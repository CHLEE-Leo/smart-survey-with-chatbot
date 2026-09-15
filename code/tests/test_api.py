import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import create_app
from src.experiment import CODE_ROOT, build_runtime, load_config
from src.llm import LLMError
from src.summary import ClosingSummary


class APITests(unittest.TestCase):
    def setUp(self):
        config = load_config(str(CODE_ROOT / "configs/smoke.yaml"))
        config["environment"]["max_turns"] = 1
        self.client = TestClient(create_app(config))

    def test_food_order_and_honest_incomplete_summary(self):
        response = self.client.post("/api/start", json={"profile": {"suspected_foods": ["계란", "우유"]}})
        self.assertEqual(response.status_code, 200)
        first = response.json()
        self.assertEqual(first["progress"]["food_id"], "계란")
        body = {"session_id": first["session_id"], "question_id": first["question_id"], "value": "집에서요"}
        second = self.client.post("/api/answer", json=body).json()
        self.assertEqual(second["progress"]["food_id"], "우유")
        self.assertEqual(self.client.post("/api/answer", json=body).status_code, 409)
        body["question_id"] = second["question_id"]
        final = self.client.post("/api/answer", json=body).json()
        self.assertTrue(final["done"])
        self.assertEqual([s["termination"] for s in final["summaries"]], ["max_turns", "max_turns"])
        self.assertNotIn("prediction", str(final))
        self.assertIn("계란:", final["message"])
        self.assertIn("우유:", final["message"])
        self.assertIn("문진을 마치겠습니다", final["message"])
        self.assertTrue(all(s["closing_summary"] for s in final["summaries"]))

    def test_no_foods_and_invalid_food_list(self):
        empty = self.client.post("/api/start", json={"profile": {"suspected_foods": []}}).json()
        self.assertTrue(empty["done"])
        self.assertEqual(empty["summaries"], [])
        response = self.client.post("/api/start", json={"profile": {"suspected_foods": ["우유", "우유"]}})
        self.assertEqual(response.status_code, 422)

    def test_summary_failure_retries_through_session_get(self):
        class RetrySummarizer:
            calls = 0

            def generate(self, observation, assessment, termination):
                self.calls += 1
                if self.calls == 1:
                    raise LLMError("summary unavailable")
                return ClosingSummary("병력이 불확실하여 추가 확인이 필요합니다.",
                                      [{"turn_id": "user_1", "quote": "모름"}])

        config = load_config(str(CODE_ROOT / "configs/smoke.yaml"))
        config["environment"]["max_turns"] = 1
        runtime = build_runtime(config)
        runtime.summarizer = RetrySummarizer()
        client = TestClient(create_app(config, runtime))
        first = client.post("/api/start", json={"profile": {"suspected_foods": ["계란"]}}).json()
        body = {"session_id": first["session_id"], "question_id": first["question_id"], "value": "모름"}
        self.assertEqual(client.post("/api/answer", json=body).status_code, 503)
        url = "/api/session?session_id=" + first["session_id"]
        final = client.get(url).json()
        self.assertTrue(final["done"])
        self.assertEqual(sum(m["role"] == "user" for m in final["summaries"][0]["messages"]), 1)
        self.assertEqual(client.get(url).json(), final)
        self.assertEqual(runtime.summarizer.calls, 2)


if __name__ == "__main__":
    unittest.main()
