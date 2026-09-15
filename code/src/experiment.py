"""Configuration, component assembly and checkpoints shared by the entry points."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import random
from pathlib import Path
from types import SimpleNamespace

import torch
import yaml

from .assessment import DemoAssessor, LLMAssessor, TARGET_VERSION
from .dialogue import LLMQuestioner, TemplateQuestioner
from .environment import FoodInterviewEnv
from .llm import HTTPTextGenerator, PROMPT_VERSION
from .model import build_tiny_policy, load_llm_policy
from .questionnaire import QUESTIONNAIRE_VERSION
from .reward import REWARD_VERSION, RewardConfig
from .summary import LLMClosingSummarizer, SUMMARY_VERSION, TemplateClosingSummarizer
from .user_simulator import LLMUserSimulator, ScriptedUserSimulator

CODE_ROOT = Path(__file__).resolve().parents[1]
VERSIONS = {"checkpoint": 1, "questionnaire": QUESTIONNAIRE_VERSION, "prompt": PROMPT_VERSION,
            "target": TARGET_VERSION, "reward": REWARD_VERSION}


def load_config(path: str) -> dict:
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    expected = {"seed", "data", "output_dir", "model", "generation", "simulator", "environment", "reward", "ppo"}
    if not isinstance(config, dict) or set(config) != expected:
        raise ValueError(f"Config top-level keys must be {sorted(expected)}")
    # Dataset/output paths are relative to code/, independent of the caller's cwd.
    for key in ("data", "output_dir"):
        config[key] = str((CODE_ROOT / config[key]).resolve())
    ppo = config["ppo"]
    expected_ppo = {"learning_rate", "updates", "sessions_per_update", "epochs", "minibatch_size",
                    "gamma", "gae_lambda", "clip_ratio", "value_coef", "entropy_coef", "max_grad_norm"}
    if set(ppo) != expected_ppo:
        raise ValueError(f"PPO keys must be {sorted(expected_ppo)}")
    for key in ("updates", "sessions_per_update", "epochs", "minibatch_size"):
        if type(ppo[key]) is not int or ppo[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not (0 <= ppo["gamma"] <= 1 and 0 <= ppo["gae_lambda"] <= 1 and 0 < ppo["clip_ratio"] < 1):
        raise ValueError("Invalid PPO discount, lambda or clipping ratio")
    if ppo["learning_rate"] <= 0 or ppo["max_grad_norm"] <= 0 or min(ppo["value_coef"], ppo["entropy_coef"]) < 0:
        raise ValueError("Invalid PPO learning rate or loss coefficient")
    return config


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_runtime(config: dict):
    model_config = dict(config["model"])
    backend = model_config.pop("backend")
    simulator_config = dict(config["simulator"])
    simulator_backend = simulator_config.pop("backend")
    if backend == "tiny":
        if simulator_backend != "scripted" or simulator_config or config["generation"]:
            raise ValueError("tiny requires scripted simulation and empty generation settings")
        model = build_tiny_policy(**model_config)
        questioner, assessor, simulator = TemplateQuestioner(), DemoAssessor(), ScriptedUserSimulator()
        summarizer = TemplateClosingSummarizer()
    elif backend == "llm":
        if simulator_backend != "http":
            raise ValueError("The LLM experiment requires an HTTP user simulator")
        generation = dict(config["generation"])
        summary_tokens = generation.pop("summary_max_tokens", 768)
        if set(generation) != {"question_max_tokens", "assessment_max_tokens"}:
            raise ValueError("Unknown generation settings")
        model = load_llm_policy(**model_config)
        questioner = LLMQuestioner(model, config["generation"]["question_max_tokens"])
        assessor = LLMAssessor(model, config["generation"]["assessment_max_tokens"])
        summarizer = LLMClosingSummarizer(model, summary_tokens)
        max_tokens = simulator_config.pop("max_tokens")
        simulator = LLMUserSimulator(HTTPTextGenerator(**simulator_config), max_tokens)
    else:
        raise ValueError(f"Unknown model backend: {backend}")
    reward = RewardConfig(**config["reward"])
    env = FoodInterviewEnv(questioner, assessor, simulator, reward, summarizer=summarizer, **config["environment"])
    return SimpleNamespace(model=model, questioner=questioner, assessor=assessor, summarizer=summarizer, env=env)


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted(CODE_ROOT.glob("src/*.py")) + [CODE_ROOT / "train.py", CODE_ROOT / "evaluate.py"]:
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def provenance(config: dict) -> dict:
    versions = {}
    for name in ("torch", "transformers", "peft", "bitsandbytes", "PyYAML"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    return {"versions": VERSIONS, "summary_version": SUMMARY_VERSION,
            "packages": versions, "source_sha256": source_hash(),
            "data_sha256": hashlib.sha256(Path(config["data"]).read_bytes()).hexdigest()}


def append_jsonl(path: Path, value: dict):
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + "\n")


def save_checkpoint(path: Path, model, optimizer, update: int, config: dict):
    if config["model"]["backend"] == "tiny":
        weights = {"model": model.state_dict()}
    else:
        from peft import get_peft_model_state_dict
        weights = {"adapter": get_peft_model_state_dict(model.backbone),
                   "policy_head": model.policy_head.state_dict(), "value_head": model.value_head.state_dict()}
    resolved_config = json.loads(json.dumps(config))
    if hasattr(model, "model_revision"):
        resolved_config["model"]["revision"] = model.model_revision
    state = {"weights": weights, "optimizer": optimizer.state_dict() if optimizer else None,
             "update": update, "config": resolved_config, "provenance": provenance(config),
             "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
             "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    torch.save(state, temp)
    temp.replace(path)


def read_checkpoint(path: str) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["provenance"]["versions"] != VERSIONS:
        raise ValueError("Checkpoint questionnaire/prompt/target/reward versions differ")
    return state


def restore_checkpoint(state: dict, model, optimizer=None, restore_rng: bool = False):
    weights = state["weights"]
    if "model" in weights:
        model.load_state_dict(weights["model"], strict=True)
    else:
        from peft import get_peft_model_state_dict, set_peft_model_state_dict
        expected = get_peft_model_state_dict(model.backbone)
        if set(weights["adapter"]) != set(expected):
            raise ValueError("Checkpoint LoRA parameters do not match the model")
        set_peft_model_state_dict(model.backbone, weights["adapter"])
        model.policy_head.load_state_dict(weights["policy_head"], strict=True)
        model.value_head.load_state_dict(weights["value_head"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
    if restore_rng:
        random.setstate(state["python_rng"])
        torch.set_rng_state(state["torch_rng"])
        if torch.cuda.is_available() and state["cuda_rng"]:
            torch.cuda.set_rng_state_all(state["cuda_rng"])
    model.eval()
