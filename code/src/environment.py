"""Public interview state and a small reset/step environment for training."""

from __future__ import annotations

import copy
import time
from dataclasses import asdict, dataclass, field

from .assessment import Assessment
from .dialogue import Question
from .questionnaire import get_question
from .reward import RewardConfig, score_terminal
from .summary import ClosingSummary, TemplateClosingSummarizer


@dataclass
class InterviewSession:
    """One food. The caller advances through D's food list in its original order."""
    public_profile: dict
    food_id: str
    questioner: object
    assessor: object
    max_turns: int = 12
    summarizer: object = field(default_factory=TemplateClosingSummarizer)
    messages: list[dict] = field(default_factory=list)
    asked_actions: list[int] = field(default_factory=list)
    pending_question: Question = None
    assessment: Assessment = None
    termination: str = None
    closing_summary: ClosingSummary = None

    def __post_init__(self):
        if type(self.max_turns) is not int or self.max_turns < 1:
            raise ValueError("max_turns must be positive")
        self.public_profile = copy.deepcopy(self.public_profile)
        if self.food_id not in self.public_profile.get("suspected_foods", []):
            raise ValueError("Current food must be selected in D")
        self.assessment = Assessment(self.food_id)

    @property
    def turn_count(self) -> int:
        return len(self.asked_actions)

    def observation(self) -> dict:
        # Internal predictions and private truth are not appended to policy input.
        return copy.deepcopy({"public_profile": self.public_profile, "food_id": self.food_id,
                              "messages": self.messages, "asked_actions": self.asked_actions,
                              "turn_count": self.turn_count})

    def ask(self, action_id: int) -> Question:
        get_question(action_id)
        if self.termination or self.pending_question:
            raise ValueError("Cannot ask after termination or before answering a pending question")
        self.pending_question = self.questioner.generate(self.observation(), action_id)
        return self.pending_question

    def answer(self, text: str) -> Assessment:
        if self.termination or self.pending_question is None:
            raise ValueError("No question awaiting an answer")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("A nonempty user answer is required")
        turn = self.turn_count + 1
        question = self.pending_question
        observation = self.observation()
        observation["messages"].extend([
            {"id": f"assistant_{turn}", "role": "assistant", "content": question.text},
            {"id": f"user_{turn}", "role": "user", "content": text.strip()},
        ])
        observation["asked_actions"].append(question.action_id)
        observation["turn_count"] = turn
        assessment = self.assessor.assess(observation)
        # Commit only after successful assessment; a retry cannot duplicate a turn.
        self.messages = observation["messages"]
        self.asked_actions = observation["asked_actions"]
        self.assessment = assessment
        self.pending_question = None
        if assessment.ready_to_stop:
            self.termination = "sufficient"
        elif turn >= self.max_turns:
            # This limit is a task terminal, not a collector time-limit truncation.
            self.termination = "max_turns"
        return assessment

    def finish(self) -> ClosingSummary:
        """Generate once after the stop decision. Failed generation can be retried.

        The answer and assessment are already committed. This output is not a
        new action, user response or input to the terminal reward decision.
        """
        if not self.termination:
            raise ValueError("Cannot summarize an ongoing food interview")
        if self.closing_summary is None:
            summary = self.summarizer.generate(self.observation(), self.assessment, self.termination)
            self.closing_summary = summary
            self.messages.append({"id": "assistant_summary", "role": "assistant", "kind": "summary",
                                  "content": summary.text})
        return self.closing_summary

    def summary(self) -> dict:
        return {"food_id": self.food_id, "turns": self.turn_count,
                "termination": self.termination, "assessment": self.assessment.to_dict(),
                "messages": copy.deepcopy(self.messages),
                "closing_summary": self.closing_summary.to_dict() if self.closing_summary else None}


class FoodInterviewEnv:
    def __init__(self, questioner, assessor, user_simulator, reward_config: RewardConfig, max_turns: int = 12,
                 summarizer=None):
        if type(max_turns) is not int or max_turns < 1:
            raise ValueError("max_turns must be a positive integer")
        self.questioner, self.assessor = questioner, assessor
        self.user_simulator, self.reward_config = user_simulator, reward_config
        self.max_turns = max_turns
        self.summarizer = summarizer or TemplateClosingSummarizer()

    def reset(self, scenario, food_id: str) -> dict:
        self.last_attempt = {"phase": "reset"}
        self.session = InterviewSession(scenario.public_profile, food_id, self.questioner,
                                        self.assessor, self.max_turns, summarizer=self.summarizer)
        self.reference = copy.deepcopy(scenario.reference[food_id])
        self.scenario_id = scenario.scenario_id
        self.user_simulator.reset(scenario, food_id)
        return self.session.observation()

    def step(self, action_id: int) -> tuple:
        start = time.perf_counter()
        self.last_attempt = {"phase": "question", "action_id": action_id}
        repeat = action_id in self.session.asked_actions
        question = self.session.ask(action_id)
        self.last_attempt.update(phase="user_simulator", question=asdict(question))
        reply = self.user_simulator.respond(self.session.observation(), question)
        self.last_attempt.update(phase="assessment", reply=reply)
        assessment = self.session.answer(reply)
        done = self.session.termination is not None
        reward = -self.reward_config.turn_cost
        metrics = {}
        if done:
            self.last_attempt["phase"] = "closing_summary"
            self.session.finish()
            terminal_reward, metrics = score_terminal(assessment, self.session.termination,
                                                       self.reference, self.reward_config)
            reward += terminal_reward
        self.last_attempt["phase"] = "complete"
        info = {"scenario_id": self.scenario_id, "food_id": self.session.food_id,
                "turn": self.session.turn_count, "question": asdict(question), "reply": reply,
                "repeated_action": repeat, "assessment": assessment.to_dict(), "reward": reward,
                "termination": self.session.termination, "elapsed_seconds": time.perf_counter() - start,
                "closing_summary": self.session.closing_summary.to_dict() if done else None,
                **metrics}
        return self.session.observation(), reward, done, False, info
