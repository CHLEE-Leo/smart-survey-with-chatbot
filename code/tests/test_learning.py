import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.experiment import (CODE_ROOT, build_runtime, load_config, read_checkpoint,
                            restore_checkpoint, save_checkpoint, seed_everything)
from src.model import ActorCritic, ByteTokenizer, build_tiny_policy, load_llm_policy, pad_tokens
from src.ppo import Transition, collect_episode, compute_gae, ppo_update
from src.user_simulator import load_scenarios

PPO = dict(epochs=2, minibatch_size=4, gamma=1.0, gae_lambda=0.95, clip_ratio=0.2,
           value_coef=0.5, entropy_coef=0.01, max_grad_norm=1.0)


class LearningTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        seed_everything(5)

    def test_terminal_reward_propagates_to_earlier_actions(self):
        advantages, returns = compute_gae([0, 0, 1], [0, 0, 0], [0, 0, 0],
                                          [False, False, True], [False] * 3, gamma=0.9, gae_lambda=1)
        torch.testing.assert_close(returns, torch.tensor([0.81, 0.9, 1.0]))

    def test_bootstrap_at_truncation_but_no_trace_across_episodes(self):
        advantages, _ = compute_gae([1, 100], [0.5, 0], [2, 999],
                                    [False, True], [True, False], gamma=0.9, gae_lambda=1)
        torch.testing.assert_close(advantages, torch.tensor([2.3, 100.0]))

    def test_padding_does_not_change_policy_log_probability(self):
        model = build_tiny_policy()
        a, b = torch.tensor([2, 3, 4]), torch.tensor([5, 6, 7, 8, 9])
        ids, mask = pad_tokens([a, b], 0, model.device)
        batch_logits, batch_values = model(ids, mask)
        logits, value = model(a[None], torch.ones_like(a[None]))
        torch.testing.assert_close(batch_logits[0], logits[0])
        torch.testing.assert_close(batch_values[0], value[0])

    def test_ppo_updates_adapter_and_heads_but_preserves_frozen_base(self):
        config = load_config(str(CODE_ROOT / "configs/smoke.yaml"))
        runtime = build_runtime(config)
        model = runtime.model
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        before = {name: p.detach().clone() for name, p in model.named_parameters()}
        scenario = load_scenarios(Path(config["data"]), "train")[0]
        episode, _ = collect_episode(model, runtime.env, scenario, "계란")
        stats = ppo_update(model, optimizer, episode, **PPO)
        self.assertTrue(all(torch.isfinite(torch.tensor(v)) for v in stats.values()))
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                self.assertTrue(torch.equal(before[name], parameter), name)
        for name in ("backbone.lora_B.weight", "policy_head.weight", "value_head.weight"):
            self.assertFalse(torch.equal(before[name], dict(model.named_parameters())[name]), name)
        self.assertTrue(all(t.tokens.device.type == "cpu" for t in episode))

    def test_checkpoint_round_trip_includes_heads_optimizer_and_rng(self):
        config = load_config(str(CODE_ROOT / "configs/smoke.yaml"))
        model = build_tiny_policy()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad])
        tokens = torch.tensor([1, 2, 3])
        a, logp, value = model.act(tokens)
        ppo_update(model, optimizer, [Transition(tokens, a, logp, value, 1, True)], **PPO)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(path, model, optimizer, 1, config)
            expected_random = torch.rand(3)
            state = read_checkpoint(str(path))
            restored = build_tiny_policy()
            restored_optimizer = torch.optim.AdamW([p for p in restored.parameters() if p.requires_grad])
            restore_checkpoint(state, restored, restored_optimizer, restore_rng=True)
            torch.testing.assert_close(torch.rand(3), expected_random)
            self.assertEqual(model.act(tokens, True), restored.act(tokens, True))
            self.assertEqual(len(optimizer.state), len(restored_optimizer.state))


@unittest.skipUnless(importlib.util.find_spec("transformers") and importlib.util.find_spec("peft"),
                     "Install Transformers and PEFT to run the real adapter integration tests")
class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        from transformers import GPT2Config, GPT2LMHeadModel
        from peft import LoraConfig, get_peft_model
        torch.set_num_threads(1)
        seed_everything(3)
        base = GPT2LMHeadModel(GPT2Config(n_layer=1, n_head=2, n_embd=16, vocab_size=257,
                                         n_positions=128, bos_token_id=1, eos_token_id=1))
        base.requires_grad_(False)
        backbone = get_peft_model(base, LoraConfig(r=2, lora_alpha=4, target_modules=["c_attn"],
                                                  fan_in_fan_out=True, task_type="CAUSAL_LM", lora_dropout=0))
        self.model = ActorCritic(backbone, ByteTokenizer(), 16, 128)

    def test_pretrained_loader_with_local_qwen_weights_and_tokenizer(self):
        from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import Whitespace
        tokenizer = Tokenizer(WordLevel({"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hello": 3}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer, pad_token="[PAD]",
                                            unk_token="[UNK]", eos_token="[EOS]")
        tokenizer.chat_template = "{% for m in messages %}{{m['role']}}: {{m['content']}}\n{% endfor %}assistant:"
        base = Qwen2ForCausalLM(Qwen2Config(vocab_size=4, hidden_size=16, intermediate_size=32,
                                          num_hidden_layers=1, num_attention_heads=2,
                                          num_key_value_heads=1, max_position_embeddings=512))
        with tempfile.TemporaryDirectory() as directory:
            base.save_pretrained(directory)
            tokenizer.save_pretrained(directory)
            model = load_llm_policy(directory, "main", quantize=False, dtype="float32", device="cpu",
                                    max_input_tokens=512, lora_r=2, lora_alpha=4)
            tokens = model.encode({"public_profile": {"suspected_foods": ["egg"]}, "food_id": "egg",
                                   "messages": [], "asked_actions": [], "turn_count": 0})
            ids, mask = pad_tokens([tokens, tokens[:5]], model.pad_id, model.device)
            logits, values = model(ids, mask)
            self.assertEqual(tuple(logits.shape), (2, 7))
            self.assertEqual(tuple(values.shape), (2,))
            self.assertTrue(any("lora_" in name and p.requires_grad for name, p in model.named_parameters()))
            self.assertIsNotNone(model.backbone.get_output_embeddings())

    def test_real_lora_gradients_and_disabled_adapter_base_stability(self):
        model = self.model
        tokens = torch.tensor([12, 13, 14])
        with torch.no_grad(), model.backbone.disable_adapter():
            before_base = model.backbone(input_ids=tokens[None]).logits.detach().clone()
        frozen = {n: p.detach().clone() for n, p in model.backbone.named_parameters() if not p.requires_grad}
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.02)
        transitions = []
        for i in range(4):
            ids = tokens + i
            action, logp, value = model.act(ids)
            transitions.append(Transition(ids, action, logp, value, float(i % 2), True))
        ppo_update(model, optimizer, transitions, **PPO)
        gradients = [p.grad for n, p in model.backbone.named_parameters() if "lora_B" in n]
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in gradients))
        for name, parameter in model.backbone.named_parameters():
            if name in frozen:
                self.assertTrue(torch.equal(frozen[name], parameter), name)
        with torch.no_grad(), model.backbone.disable_adapter():
            after_base = model.backbone(input_ids=tokens[None]).logits
        torch.testing.assert_close(before_base, after_base, rtol=0, atol=0)

    def test_generation_disables_adapter_and_restores_it_even_on_error(self):
        model = self.model
        from peft.tuners.lora import LoraLayer
        layers = [m for m in model.backbone.modules() if isinstance(m, LoraLayer)]

        def fail(**kwargs):
            self.assertTrue(all(layer.disable_adapters for layer in layers))
            raise RuntimeError("test failure")

        with patch.object(model.backbone, "generate", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "test failure"):
                model.complete([{"role": "user", "content": "hello"}], max_tokens=4)
        self.assertTrue(all(not layer.disable_adapters for layer in layers))

    def test_real_adapter_checkpoint_round_trip(self):
        config = load_config(str(CODE_ROOT / "configs/smoke.yaml"))
        config["model"] = {"backend": "llm", "name": "offline-gpt2-test", "revision": "test"}
        model = self.model
        tokens = torch.tensor([4, 5, 6])
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
        action, logp, value = model.act(tokens)
        ppo_update(model, optimizer, [Transition(tokens, action, logp, value, 1, True)], **PPO)
        expected = model.act(tokens, True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter.pt"
            save_checkpoint(path, model, optimizer, 1, config)
            state = read_checkpoint(str(path))
            self.assertEqual(set(state["weights"]), {"adapter", "policy_head", "value_head"})
            self.setUp()  # Same frozen base seed; adapters and both heads are restored.
            restore_checkpoint(state, self.model)
            self.assertEqual(expected, self.model.act(tokens, True))


if __name__ == "__main__":
    unittest.main()
