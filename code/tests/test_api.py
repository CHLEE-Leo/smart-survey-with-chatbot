import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.testclient import TestClient

from api.main import create_app
from src.experiment import CODE_ROOT, load_config


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

    def test_no_foods_and_invalid_food_list(self):
        empty = self.client.post("/api/start", json={"profile": {"suspected_foods": []}}).json()
        self.assertTrue(empty["done"])
        self.assertEqual(empty["summaries"], [])
        response = self.client.post("/api/start", json={"profile": {"suspected_foods": ["우유", "우유"]}})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
