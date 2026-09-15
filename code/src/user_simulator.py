"""Scenario data and natural-language user simulation; labels stay in reward.py."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path

from .assessment import LABELS, VALUES
from .llm import LLMError, TextGenerator, public_context


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    group_id: str
    split: str
    public_profile: dict
    private_history: dict
    persona: str
    reference: dict
    demo_answers: dict = field(default_factory=dict)


def load_scenarios(path: Path, split: str) -> list[Scenario]:
    scenarios, ids, groups = [], set(), {}
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            scenario = Scenario(**json.loads(line))
            foods = scenario.public_profile["suspected_foods"]
            if (not isinstance(foods, list) or not foods or len(set(foods)) != len(foods)
                    or not all(isinstance(x, str) and x.strip() for x in foods)):
                raise ValueError("Scenario needs unique, nonempty food names")
            if scenario.scenario_id in ids:
                raise ValueError("Duplicate scenario_id")
            if scenario.group_id in groups and groups[scenario.group_id] != scenario.split:
                raise ValueError("Patient/group appears in multiple splits")
            if scenario.split not in ("train", "validation", "test"):
                raise ValueError("Unknown split")
            for food in foods:
                ref = scenario.reference[food]
                if ref["label"] not in LABELS:
                    raise ValueError("Unknown reference label")
                required = ref["required_evidence"]
                if not isinstance(required, dict) or (ref["label"] != "undetermined" and not required):
                    raise ValueError("Decidable labels need independently specified evidence")
                if any(k not in VALUES or v not in VALUES[k] for k, v in required.items()):
                    raise ValueError("Invalid required evidence")
                if food not in scenario.private_history:
                    raise ValueError("Missing private food history")
            ids.add(scenario.scenario_id)
            groups[scenario.group_id] = scenario.split
            scenarios.append(scenario)
        except (TypeError, KeyError, ValueError) as exc:
            raise ValueError(f"Invalid scenario at {path}:{line_no}: {exc}") from exc
    selected = [s for s in scenarios if s.split == split]
    if not selected:
        raise ValueError(f"No scenarios in split {split}")
    return selected


class LLMUserSimulator:
    def __init__(self, generator: TextGenerator, max_tokens: int = 256):
        self.generator = generator
        self.max_tokens = max_tokens

    def reset(self, scenario: Scenario, food_id: str):
        # Deliberately keep neither the Scenario nor its reference in this object.
        self.private_history = copy.deepcopy(scenario.private_history[food_id])
        self.persona = scenario.persona

    def respond(self, observation: dict, question) -> str:
        prompt = (
            "당신은 문진에 답하는 보호자다. 고정된 이력과 기억 설정에 맞게 자연스러운 한국어로 답한다. "
            "바로 아래 실제 질문에 답하고, 맥락상 자연스러우면 여러 사실을 함께 말할 수 있다. "
            "질문자의 추측에 맞추어 사실을 만들거나 바꾸지 않는다. 모르는 것은 모른다고 한다. "
            "진단 정답을 추측해 발표하지 않는다. 다음은 보호자가 실제로 알고 있는 이력이다:\n"
            + json.dumps({"history": self.private_history, "persona": self.persona}, ensure_ascii=False))
        return self.generator.complete([
            {"role": "system", "content": prompt},
            {"role": "user", "content": public_context(observation) + "\n실제 질문: " + question.text},
        ], max_tokens=self.max_tokens)


class ScriptedUserSimulator:
    """Deterministic action-indexed test double; not an LLM/user realism model."""
    def reset(self, scenario: Scenario, food_id: str):
        self.answers = copy.deepcopy(scenario.demo_answers[food_id])
        self.counts = {}

    def respond(self, observation: dict, question) -> str:
        key = str(question.action_id)
        answers = self.answers.get(key)
        if not isinstance(answers, list) or not answers or not all(isinstance(x, str) and x for x in answers):
            raise LLMError("Missing scripted answer for an action")
        index = self.counts.get(key, 0)
        self.counts[key] = index + 1
        return answers[min(index, len(answers) - 1)]
