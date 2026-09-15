import copy
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.assessment import DemoAssessor, LLMAssessor
from src.dialogue import LLMQuestioner, Question, TemplateQuestioner
from src.environment import FoodInterviewEnv, InterviewSession
from src.llm import LLMError, public_context
from src.questionnaire import NUM_ACTIONS, SYMPTOMS, latency_bin
from src.reward import RewardConfig
from src.summary import LLMClosingSummarizer
from src.user_simulator import LLMUserSimulator, ScriptedUserSimulator, load_scenarios

DATA = Path(__file__).resolve().parents[1] / "data/demo_scenarios.jsonl"


class FakeGenerator:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    def complete(self, messages, *, max_tokens):
        self.calls.append(messages)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class InterviewTests(unittest.TestCase):
    def setUp(self):
        self.scenarios = load_scenarios(DATA, "train")
        self.env = FoodInterviewEnv(TemplateQuestioner(), DemoAssessor(), ScriptedUserSimulator(), RewardConfig())

    def test_source_action_count_and_latency_boundaries(self):
        self.assertEqual((NUM_ACTIONS, len(SYMPTOMS)), (7, 18))
        self.assertEqual([latency_bin(x) for x in (0, 29.9, 30, 119.9, 120, 239.9, 240)],
                         ["lt30", "lt30", "30to120", "30to120", "120to240", "120to240", "ge240"])
        for x in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                latency_bin(x)

    def test_clarification_repeats_same_action_and_enables_stop(self):
        scenario = self.scenarios[1]
        self.env.reset(scenario, "우유")
        for action in (1, 2, 3, 5):
            _, _, done, _, _ = self.env.step(action)
            self.assertFalse(done)
        _, reward, done, truncated, info = self.env.step(2)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])
        self.assertTrue(info["repeated_action"])
        self.assertGreater(reward, 0)

    def test_changing_gold_changes_reward_not_observations_or_stop(self):
        scenario = self.scenarios[0]
        wrong = copy.deepcopy(scenario.reference)
        wrong["계란"]["label"] = "not_meets_criteria"
        traces, returns = [], []
        for case in (scenario, replace(scenario, reference=wrong)):
            trace = [self.env.reset(case, "계란")]
            for action in (1, 2, 3, 5):
                observation, reward, done, _, _ = self.env.step(action)
                trace.append((observation, done, self.env.session.termination))
            traces.append(trace)
            returns.append(reward)
        self.assertEqual(traces[0], traces[1])
        self.assertGreater(returns[0], 0)
        self.assertLess(returns[1], 0)

    def test_turn_limit_is_incomplete_terminal(self):
        self.env.max_turns = 1
        self.env.reset(self.scenarios[0], "계란")
        _, reward, done, truncated, info = self.env.step(0)
        self.assertTrue(done)
        self.assertFalse(truncated)
        self.assertEqual(info["termination"], "max_turns")
        self.assertFalse(info["success"])
        self.assertLess(reward, 0)

    def test_volunteered_facts_can_complete_without_visiting_every_action(self):
        session = InterviewSession({"suspected_foods": ["계란"]}, "계란", TemplateQuestioner(), DemoAssessor())
        session.ask(1)
        session.answer("증상: 두드러기. 발현 시간: 20분. 반복: 재섭취 안 함. 현재 섭취: 제한 중")
        self.assertEqual(session.termination, "sufficient")
        self.assertEqual(session.turn_count, 1)

    def test_assessment_rejects_invented_or_assistant_evidence(self):
        observation = self.env.reset(self.scenarios[0], "계란")
        observation["messages"] = [{"id": "user_1", "role": "user", "content": "모르겠어요"},
                                   {"id": "assistant_1", "role": "assistant", "content": "두드러기인가요?"}]
        for turn_id, quote in (("user_1", "두드러기"), ("assistant_1", "두드러기")):
            result = {"evidence": {"symptoms": {"value": "typical", "turn_id": turn_id, "quote": quote}},
                      "contradictions": []}
            with self.assertRaises(LLMError):
                LLMAssessor(FakeGenerator([json.dumps(result)])).assess(observation)

    def test_unknown_and_conflicting_evidence_do_not_become_negative(self):
        session = InterviewSession({"suspected_foods": ["계란"]}, "계란", TemplateQuestioner(), DemoAssessor())
        session.ask(2)
        session.answer("전혀 기억나지 않아요")
        self.assertEqual(session.assessment.prediction, "undetermined")
        session.ask(5)
        session.answer("현재 섭취: 제한 중. 현재 섭취: 문제없이 먹음")
        self.assertFalse(session.assessment.ready_to_stop)
        self.assertEqual(session.assessment.contradictions, ["current_intake"])

    def test_failed_assessment_keeps_answer_retryable(self):
        assessor = LLMAssessor(FakeGenerator([LLMError("offline"), '{"evidence": {}, "contradictions": []}']))
        session = InterviewSession({"suspected_foods": ["계란"]}, "계란", TemplateQuestioner(), assessor)
        session.ask(1)
        with self.assertRaises(LLMError):
            session.answer("모르겠어요")
        self.assertEqual(session.messages, [])
        self.assertIsNotNone(session.pending_question)
        session.answer("모르겠어요")
        self.assertEqual(session.turn_count, 1)

    def test_simulator_receives_actual_question_but_no_reference(self):
        scenario = self.scenarios[0]
        reference = {"계란": {"label": "PRIVATE_GOLD_SENTINEL", "required_evidence": {}}}
        generator = FakeGenerator(["모르겠어요"])
        simulator = LLMUserSimulator(generator)
        simulator.reset(replace(scenario, reference=reference), "계란")
        observation = self.env.reset(scenario, "계란")
        question = Question(2, "목욕하기 전까지 얼마나 시간이 있었나요?")
        simulator.respond(observation, question)
        prompt = json.dumps(generator.calls, ensure_ascii=False)
        self.assertIn(question.text, prompt)
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", prompt)
        observation["reference"] = reference
        self.assertNotIn("PRIVATE_GOLD_SENTINEL", public_context(observation))

    def test_invalid_generated_question_uses_logged_skeleton_fallback(self):
        generator = FakeGenerator(["증상과 시간과 검사와 현재 섭취를 모두 말해 주세요.",
                                   '{"valid": false, "reason": "multiple topics"}'])
        observation = self.env.reset(self.scenarios[0], "계란")
        question = LLMQuestioner(generator).generate(observation, 2)
        self.assertTrue(question.fallback_used)
        self.assertIn("얼마나", question.text)
        self.assertEqual(question.generation_calls, 2)

    def test_patient_group_cannot_cross_dataset_splits(self):
        rows = [json.loads(line) for line in DATA.read_text().splitlines()]
        rows[-1]["group_id"] = rows[0]["group_id"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "multiple splits"):
                load_scenarios(path, "train")

    def test_closing_summary_is_cached_and_does_not_add_an_action(self):
        note = "계란을 집에서 섭취했다고 보고했습니다. 증상과 반응 시간은 확인되지 않아 판단을 보류합니다."
        payload = json.dumps({"text": note, "citations": [{"turn_id": "user_1", "quote": "집에서요"}]})
        generator = FakeGenerator([payload, '{"valid": true, "reason": "원문과 일치"}'])
        session = InterviewSession({"suspected_foods": ["계란"]}, "계란", TemplateQuestioner(), DemoAssessor(),
                                   max_turns=1, summarizer=LLMClosingSummarizer(generator))
        with self.assertRaises(ValueError):
            session.finish()
        session.ask(0)
        session.answer("집에서요")
        summary = session.finish()
        self.assertIs(summary, session.finish())
        self.assertEqual(len(generator.calls), 2)
        self.assertEqual(session.asked_actions, [0])
        self.assertEqual(session.messages[-1]["kind"], "summary")
        self.assertEqual(session.messages[-1]["content"], note)
        self.assertEqual(sum(m.get("kind") == "summary" for m in session.messages), 1)
        self.assertEqual(session.assessment.prediction, "undetermined")
        prompt = generator.calls[0][1]["content"]
        self.assertEqual(json.loads(prompt)["termination"], "max_turns")

    def test_closing_summary_does_not_receive_private_truth(self):
        observation = self.env.reset(self.scenarios[0], "계란")
        observation["messages"] = [{"id": "user_1", "role": "user", "content": "기억나지 않아요"}]
        observation["reference"] = "PRIVATE_REFERENCE"
        observation["private_history"] = "PRIVATE_HISTORY"
        generator = FakeGenerator([
            '{"text": "기억이 불확실하여 판단을 보류합니다.", "citations": [{"turn_id": "user_1", "quote": "기억나지 않아요"}]}',
            '{"valid": true, "reason": "기억 불확실"}',
        ])
        LLMClosingSummarizer(generator).generate(observation, self.env.session.assessment, "max_turns")
        prompts = json.dumps(generator.calls)
        self.assertNotIn("PRIVATE_REFERENCE", prompts)
        self.assertNotIn("PRIVATE_HISTORY", prompts)

    def test_summary_rejects_fabricated_citation(self):
        observation = self.env.reset(self.scenarios[0], "계란")
        observation["messages"] = [{"id": "user_1", "role": "user", "content": "기억나지 않아요"}]
        generator = FakeGenerator([
            '{"text": "입술 부종이 있었다고 합니다.", "citations": [{"turn_id": "user_1", "quote": "입술 부종"}]}',
        ])
        with self.assertRaises(LLMError):
            LLMClosingSummarizer(generator).generate(observation, self.env.session.assessment, "max_turns")

    def test_rejected_summary_can_retry_without_repeating_user_answer(self):
        bad = '{"text": "알레르기가 없습니다.", "citations": [{"turn_id": "user_1", "quote": "모름"}]}'
        good = '{"text": "기억이 불확실하여 알레르기 판단을 보류합니다.", "citations": [{"turn_id": "user_1", "quote": "모름"}]}'
        generator = FakeGenerator([bad, '{"valid": false, "reason": "부재를 확진함"}',
                                   good, '{"valid": true, "reason": "보류를 유지함"}'])
        session = InterviewSession({"suspected_foods": ["계란"]}, "계란", TemplateQuestioner(), DemoAssessor(),
                                   max_turns=1, summarizer=LLMClosingSummarizer(generator))
        session.ask(1)
        session.answer("모름")
        with self.assertRaises(LLMError):
            session.finish()
        self.assertIsNone(session.closing_summary)
        self.assertEqual(len(session.messages), 2)
        self.assertEqual(session.termination, "max_turns")
        session.finish()
        self.assertEqual(len(session.messages), 3)
        self.assertEqual(session.turn_count, 1)


if __name__ == "__main__":
    unittest.main()
