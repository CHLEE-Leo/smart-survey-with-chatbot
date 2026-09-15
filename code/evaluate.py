"""Evaluate a frozen checkpoint or simple question-order baselines on held-out scenarios."""

import argparse
import json
import random
from pathlib import Path

from src.experiment import (CODE_ROOT, append_jsonl, build_runtime, load_config, provenance,
                            read_checkpoint, restore_checkpoint, seed_everything)
from src.user_simulator import load_scenarios


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CODE_ROOT / "configs/default.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy", choices=("learned", "sequential", "random"), default="learned")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.policy == "learned" and not args.checkpoint:
        parser.error("--policy learned requires --checkpoint")
    output = Path(args.output)
    if output.exists():
        parser.error("Evaluation output already exists; choose a new file")
    output.parent.mkdir(parents=True, exist_ok=True)
    state = read_checkpoint(args.checkpoint) if args.checkpoint else None
    config = state["config"] if state else load_config(args.config)
    scenarios = load_scenarios(Path(config["data"]), args.split)
    seed_everything(args.seed)
    runtime = build_runtime(config)
    if state:
        restore_checkpoint(state, runtime.model)
    results = []
    for scenario in scenarios:
        for food in scenario.public_profile["suspected_foods"]:
            logs = []
            try:
                observation = runtime.env.reset(scenario, food)
                while True:
                    if args.policy == "learned":
                        action = runtime.model.act(runtime.model.encode(observation), deterministic=True)[0]
                    elif args.policy == "sequential":
                        action = observation["turn_count"] % 7
                    else:
                        action = random.randrange(7)
                    observation, _, done, _, info = runtime.env.step(action)
                    logs.append(info)
                    if done:
                        break
                result = {"scenario_id": scenario.scenario_id, "food_id": food,
                          "turns": logs, "outcome": logs[-1], "error": None}
            except Exception as exc:
                result = {"scenario_id": scenario.scenario_id, "food_id": food,
                          "turns": logs, "outcome": {}, "error": str(exc),
                          "attempt": runtime.env.last_attempt}
            results.append(result)
            append_jsonl(output, result)
    n = len(results)
    summary = {"policy": args.policy, "split": args.split, "seed": args.seed, "episodes": n,
               "errors": sum(r["error"] is not None for r in results),
               "success_rate": sum(r["outcome"].get("success", False) for r in results) / n,
               "label_accuracy": sum(r["outcome"].get("label_correct", False) for r in results) / n,
               "false_sufficient_rate": sum(r["outcome"].get("false_sufficient", False) for r in results) / n,
               "mean_turns": sum(len(r["turns"]) for r in results) / n,
               "session_success_rate": sum(all(r["outcome"].get("success", False) for r in results
                                                if r["scenario_id"] == s.scenario_id) for s in scenarios) / len(scenarios),
               "offline_test_double": config["model"]["backend"] == "tiny",
               "provenance": provenance(config)}
    summary_path = output.with_suffix(output.suffix + ".summary.json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
