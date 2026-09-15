"""Train the seven-topic policy with PPO; see README.md for real and offline runs."""

import argparse
import json
import random
from pathlib import Path

import torch

from src.experiment import (CODE_ROOT, append_jsonl, build_runtime, load_config, provenance,
                            read_checkpoint, restore_checkpoint, save_checkpoint, seed_everything)
from src.ppo import collect_episode, ppo_update
from src.user_simulator import load_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CODE_ROOT / "configs/default.yaml"))
    parser.add_argument("--output-dir")
    parser.add_argument("--updates", type=int, help="Total update count, including resumed updates")
    parser.add_argument("--resume", help="Resume config, weights, optimizer and RNG from this checkpoint")
    args = parser.parse_args()
    state = read_checkpoint(args.resume) if args.resume else None
    config = state["config"] if state else load_config(args.config)
    if args.updates is not None:
        if args.updates < 1:
            parser.error("--updates must be positive")
        config["ppo"]["updates"] = args.updates
    if args.output_dir:
        config["output_dir"] = str(Path(args.output_dir).resolve())
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not state and any(output.iterdir()):
        parser.error("Output directory is not empty; choose a new directory or use --resume")
    scenarios = load_scenarios(Path(config["data"]), "train")
    if state and provenance(config)["data_sha256"] != state["provenance"]["data_sha256"]:
        parser.error("Resume dataset differs from the checkpoint")
    seed_everything(config["seed"])
    runtime = build_runtime(config)
    model = runtime.model
    ppo = dict(config["ppo"])
    updates, sessions_per_update = ppo.pop("updates"), ppo.pop("sessions_per_update")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=ppo.pop("learning_rate"), weight_decay=0.0)
    start = state["update"] if state else 0
    if start >= updates:
        parser.error("--updates must exceed the checkpoint update count")
    if state:
        restore_checkpoint(state, model, optimizer, restore_rng=True)
    manifest = {"config": config, **provenance(config),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "offline_test_double": config["model"]["backend"] == "tiny"}
    (output / "run.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "trainable_parameters": manifest["trainable_parameters"],
                      "offline_test_double": manifest["offline_test_double"]}), flush=True)
    for update in range(start + 1, updates + 1):
        transitions, terminals = [], []
        for scenario in random.choices(scenarios, k=sessions_per_update):
            # D's order is fixed. Each food is its own episode, always with seven actions.
            for food in scenario.public_profile["suspected_foods"]:
                try:
                    episode, logs = collect_episode(model, runtime.env, scenario, food)
                except Exception as exc:
                    append_jsonl(output / "errors.jsonl", {"update": update, "scenario_id": scenario.scenario_id,
                                                           "food_id": food, "error": str(exc),
                                                           "attempt": getattr(runtime.env, "last_attempt", {}),
                                                           "observation": runtime.env.session.observation()})
                    raise  # Failed services are not synthetic patient outcomes.
                transitions.extend(episode)
                terminals.append(logs[-1])
                append_jsonl(output / "rollouts.jsonl", {"update": update, "turns": logs})
        metrics = ppo_update(model, optimizer, transitions, **ppo)
        metrics.update(update=update, episodes=len(terminals), transitions=len(transitions),
                       mean_return=sum(t.reward for t in transitions) / len(terminals),
                       success_rate=sum(t["success"] for t in terminals) / len(terminals))
        append_jsonl(output / "metrics.jsonl", metrics)
        save_checkpoint(output / "last.pt", model, optimizer, update, config)
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
