"""Training-only terminal comparison. This module never decides when to stop."""

from dataclasses import dataclass

REWARD_VERSION = "terminal-evidence-v1"


@dataclass
class RewardConfig:
    correct: float = 1.0
    incorrect: float = -1.0
    incomplete: float = -0.2
    turn_cost: float = 0.02

    def __post_init__(self):
        import math
        if not all(math.isfinite(v) for v in (self.correct, self.incorrect, self.incomplete, self.turn_cost)):
            raise ValueError("Reward values must be finite")
        if self.turn_cost < 0 or self.correct <= max(self.incorrect, self.incomplete):
            raise ValueError("Use a nonnegative cost and a greater reward for success")


def score_terminal(assessment, termination: str, reference: dict, config: RewardConfig) -> tuple:
    label_correct = assessment.prediction == reference["label"]
    evidence_correct = all(
        key in assessment.evidence and assessment.evidence[key].value == value
        for key, value in reference["required_evidence"].items())
    success = (termination == "sufficient" and assessment.ready_to_stop
               and not assessment.contradictions and label_correct and evidence_correct
               and assessment.prediction != "undetermined")
    if termination != "sufficient":
        reward = config.incomplete
    else:
        reward = config.correct if success else config.incorrect
    return reward, {"success": success, "label_correct": label_correct,
                    "evidence_correct": evidence_correct,
                    "false_sufficient": termination == "sufficient" and not success}
