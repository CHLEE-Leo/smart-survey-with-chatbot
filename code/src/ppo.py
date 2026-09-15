"""Plain PyTorch PPO: complete-episode rollouts, GAE, minibatch updates."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.distributions import Categorical

from .model import pad_tokens


@dataclass
class Transition:
    tokens: torch.Tensor
    action: int
    old_log_prob: float
    value: float
    reward: float
    terminated: bool
    truncated: bool = False
    next_value: float = 0.0


def compute_gae(rewards, values, next_values, terminated, truncated,
                gamma: float = 0.99, gae_lambda: float = 0.95) -> tuple:
    """Bootstrap a collection truncation, but never a true task terminal.

    Both boundaries cut the GAE trace so it cannot leak into the next episode.
    next_values at truncation must come from the final observation, not reset().
    """
    rewards, values, next_values = [torch.as_tensor(x, dtype=torch.float32) for x in (rewards, values, next_values)]
    terminated, truncated = [torch.as_tensor(x, dtype=torch.bool) for x in (terminated, truncated)]
    if not all(x.shape == rewards.shape for x in (values, next_values, terminated, truncated)):
        raise ValueError("GAE inputs must have equal shapes")
    advantages = torch.zeros_like(rewards)
    carry = torch.tensor(0.0)
    for t in reversed(range(len(rewards))):
        bootstrap = 0.0 if terminated[t] else next_values[t]
        delta = rewards[t] + gamma * bootstrap - values[t]
        continuation = not bool(terminated[t] or truncated[t])
        carry = delta + gamma * gae_lambda * continuation * carry
        advantages[t] = carry
    return advantages, advantages + values


def collect_episode(model, env, scenario, food_id: str, deterministic: bool = False):
    observation = env.reset(scenario, food_id)
    transitions, logs = [], []
    while True:
        tokens = model.encode(observation)
        action, log_prob, value = model.act(tokens, deterministic)
        observation, reward, terminated, truncated, info = env.step(action)
        transitions.append(Transition(tokens.cpu(), action, log_prob, value, reward, terminated, truncated))
        logs.append(info)
        if terminated or truncated:
            if truncated and not terminated:
                transitions[-1].next_value = model.act(model.encode(observation), True)[2]
            break
    for current, following in zip(transitions, transitions[1:]):
        current.next_value = following.value
    return transitions, logs


def ppo_update(model, optimizer, transitions: list[Transition], *, epochs: int,
               minibatch_size: int, gamma: float, gae_lambda: float, clip_ratio: float,
               value_coef: float, entropy_coef: float, max_grad_norm: float) -> dict:
    if not transitions:
        raise ValueError("Cannot update PPO with an empty rollout")
    advantages, returns = compute_gae(
        [t.reward for t in transitions], [t.value for t in transitions],
        [t.next_value for t in transitions], [t.terminated for t in transitions],
        [t.truncated for t in transitions], gamma, gae_lambda)
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)
    stats = []
    # Keep dropout disabled in rollout and update; gradients remain enabled.
    model.eval()
    for _ in range(epochs):
        for indices in torch.randperm(len(transitions)).split(minibatch_size):
            batch = [transitions[i] for i in indices.tolist()]
            ids, mask = pad_tokens([t.tokens for t in batch], model.pad_id, model.device)
            actions = torch.tensor([t.action for t in batch], device=model.device)
            old_log_probs = torch.tensor([t.old_log_prob for t in batch], device=model.device)
            adv, targets = advantages[indices].to(model.device), returns[indices].to(model.device)
            # Recompute from the exact observation tokens: no detached hidden cache.
            with model.lock:
                logits, values = model(ids, mask)
                distribution = Categorical(logits=logits)
                log_ratio = distribution.log_prob(actions) - old_log_probs
                ratio = log_ratio.exp()
                policy_loss = -torch.minimum(ratio * adv, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv).mean()
                value_loss = 0.5 * (values - targets).square().mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite PPO loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], max_grad_norm, error_if_nonfinite=True)
                optimizer.step()
            stats.append({"loss": loss.item(), "policy_loss": policy_loss.item(),
                          "value_loss": value_loss.item(), "entropy": entropy.item(),
                          "approx_kl": ((ratio - 1) - log_ratio).mean().item(),
                          "clip_fraction": ((ratio - 1).abs() > clip_ratio).float().mean().item(),
                          "grad_norm": float(grad_norm)})
    return {key: sum(row[key] for row in stats) / len(stats) for key in stats[0]}
