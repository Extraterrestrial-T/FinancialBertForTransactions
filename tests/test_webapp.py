"""Integration checks for the public Flask research-site API."""

from __future__ import annotations

from io import BytesIO
import unittest

try:
    from webapp.server import app
except ModuleNotFoundError:  # Flask is intentionally an optional web dependency.
    app = None


@unittest.skipIf(app is None, "Flask is not installed; install .[web] to run web tests")
class WebExplorerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = app.test_client()  # type: ignore[union-attr]

    def test_bootstrap_discovers_release_contract(self) -> None:
        response = self.client.get("/api/bootstrap")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["base_checkpoint"], "best.pt")
        self.assertEqual(len(payload["adapters"]), 6)

    def test_training_history_comes_from_committed_metrics(self) -> None:
        response = self.client.get("/api/training-history")
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(len(payload["epochs"]), 300)
        self.assertLess(payload["best_validation_loss"], payload["epochs"][0]["validation_loss"])

    def test_research_pages_and_inference_lab_are_separate_routes(self) -> None:
        routes = {
            "/": "PRAGMA-lite",
            "/architecture": "Three encoders",
            "/methodology": "How the model",
            "/results": "Two uses",
            "/lab": "Account history",
            "/upload": "Score one transaction",
        }
        for route, expected_text in routes.items():
            response = self.client.get(route)
            self.assertEqual(response.status_code, 200)
            self.assertIn(expected_text, response.get_data(as_text=True))

    def test_future_value_prediction_keeps_future_events_separate(self) -> None:
        bootstrap = self.client.get("/api/bootstrap").get_json()
        adapter = next(
            item
            for item in bootstrap["adapters"]
            if item["task"] == "future_value" and item["rank"] == 4
        )
        account = self.client.get("/api/accounts?task=future_value").get_json()["accounts"][0]
        self.assertEqual(account["split"], "test")
        cutoffs = self.client.get(
            "/api/cutoffs",
            query_string={"task": "future_value", "account_id": account["account_id"]},
        ).get_json()["cutoffs"]
        response = self.client.get(
            "/api/prediction",
            query_string={
                "adapter": adapter["id"],
                "account_id": account["account_id"],
                "cutoff": cutoffs[-1],
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["task"], "future_value")
        self.assertGreater(len(payload["history"]), 10)
        self.assertGreater(len(payload["future"]), 0)
        self.assertTrue(all(event["model_visible"] is False for event in payload["future"]))
        self.assertIsNotNone(payload["outcome"]["prediction_log1p"])

        start = payload["history"][-10]["date"]
        window_response = self.client.get(
            "/api/prediction",
            query_string={
                "adapter": adapter["id"],
                "account_id": account["account_id"],
                "cutoff": cutoffs[-1],
                "start": start,
            },
        )
        self.assertEqual(window_response.status_code, 200)
        window_payload = window_response.get_json()
        self.assertLess(window_payload["history_event_count"], payload["history_event_count"])
        self.assertEqual(window_payload["window_start"], start)

    def test_upload_scores_a_schema_aligned_transaction_csv(self) -> None:
        bootstrap = self.client.get("/api/bootstrap").get_json()
        adapter = next(
            item
            for item in bootstrap["adapters"]
            if item["task"] == "future_value" and item["rank"] == 4
        )
        csv = (
            "trans_date,amount,balance,trans_type,operation,category\n"
            "2017-01-01,100,1100,C,CIC,IN\n"
            "2017-01-04,-50,1050,D,CCW,HH\n"
            "2017-01-11,20,1070,C,CIC,IN\n"
        ).encode()
        response = self.client.post(
            "/api/upload-prediction",
            data={"adapter": adapter["id"], "file": (BytesIO(csv), "history.csv")},
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["input"]["event_count"], 3)
        self.assertIsNotNone(payload["prediction"]["prediction_log1p"])


if __name__ == "__main__":
    unittest.main()
