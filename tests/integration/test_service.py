from http.client import HTTPConnection
import json
import sqlite3
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from regime.domain import ContractError
from regime.service import dispatch, server
from regime.store import publish, reader
from regime.workflow import run
from tests.common import FAST, SOURCE, TEST, TRAIN


class Integration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.db = Path(cls.tmp.name) / "history.sqlite3"
        cls.cfg = Path(cls.tmp.name) / "settings.json"
        cls.cfg.write_text(json.dumps(asdict(FAST)))
        cls.report = run(SOURCE, cls.cfg, cls.db)
        cls.base = "/v1/runs/" + cls.report["run_id"]
        cls.token = "local-test-token-1234"

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def get(self, path):
        return dispatch(self.db, self.token, "Bearer " + self.token, path)

    def test_point_in_time_pagination_and_feature_api(self):
        instant = TEST[4].available_at
        code, page1 = self.get(self.base + "/history?limit=2&as_of=" + instant)
        self.assertEqual(code, 200)
        self.assertEqual([r["available_at"] for r in page1["items"]], [b.available_at for b in TEST[:2]])
        code, page2 = self.get(self.base + "/features?after=" + page1["next_after"] + "&as_of=" + instant)
        self.assertEqual(code, 200)
        self.assertEqual([r["available_at"] for r in page2["items"]], [b.available_at for b in TEST[2:5]])
        self.assertIsNone(page2["next_after"])
        self.assertNotIn("hmm", page2["items"][0])
        self.assertEqual(self.get(self.base + "/history?as_of=" + TEST[0].event_at)[1]["items"], [])
        self.assertEqual(self.get(self.base + "/report")[1], self.report)
        self.assertEqual(self.get(self.base + "/metadata")[1]["settings"], asdict(FAST))

    def test_fail_closed(self):
        for token in (None, "Bearer wrong"):
            self.assertEqual(dispatch(self.db, self.token, token, self.base + "/history")[0], 401)
        for query in ("limit=0", "limit=1000", "as_of=yesterday", "limit=1&limit=2", "oops=yes"):
            self.assertEqual(self.get(self.base + "/features?" + query)[0], 400)
        with self.assertRaises(ContractError):
            server(self.db, "", 0)
        self.assertEqual(dispatch(self.db.with_suffix(".absent"), self.token, None, "/health")[0], 503)
        self.assertEqual(self.get("/v1/runs/" + "0"*64 + "/history")[0], 404)

    def test_truncated_dataset_preserves_model_and_decision_prefix(self):
        prefix = Path(self.tmp.name) / "prefix.csv"
        source = SOURCE.read_bytes().splitlines(keepends=True)
        prefix.write_bytes(b"".join(source[:1 + len(TRAIN) + 20]))
        db = prefix.with_suffix(".sqlite3")
        cut = run(prefix, self.cfg, db)
        self.assertNotEqual(cut["source_sha256"], self.report["source_sha256"])
        self.assertEqual(cut["model_sha256"], self.report["model_sha256"])
        self.assertEqual(cut["holdout_bars"], 20)
        with closing(reader(db)) as part:
            rows = [json.loads(r[0]) for r in part.execute("SELECT body FROM history ORDER BY available_at")]
        self.assertEqual(rows, self.get(self.base + "/history?limit=20")[1]["items"])

    def test_concurrent_replay_is_idempotent(self):
        db = Path(self.tmp.name) / "replay.sqlite3"
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda _: run(SOURCE, self.cfg, db), range(2)))
        self.assertEqual(reports, [self.report, self.report])
        with closing(reader(db)) as copy, closing(reader(self.db)) as original:
            sql = "SELECT body FROM history ORDER BY available_at"
            self.assertEqual([r[0] for r in copy.execute(sql)], [r[0] for r in original.execute(sql)])
            self.assertEqual(copy.execute("SELECT COUNT(*) FROM history").fetchone()[0], len(TEST))
            self.assertEqual(copy.execute("SELECT COUNT(*) FROM runs").fetchone()[0], 1)

    def test_interrupted_publish_rolls_back(self):
        db = Path(self.tmp.name) / "interrupted.sqlite3"
        row = {"available_at": TEST[0].available_at}
        with self.assertRaises(sqlite3.IntegrityError):
            publish(db, {"run_id": "test"}, {}, [row, row])
        publish(db, {"run_id": "test"}, {}, [row])
        with closing(reader(db)) as conn:
            self.assertEqual(conn.execute("SELECT COUNT(*) FROM history").fetchone()[0], 1)

    def test_http_transport(self):
        with server(self.db, self.token, 0) as http:
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                with closing(HTTPConnection("127.0.0.1", http.server_port, timeout=3)) as conn:
                    conn.request("GET", "/health")
                    response = conn.getresponse()
                    self.assertEqual((response.status, json.load(response)), (200, {"ready": True}))
                    conn.request("GET", "/v1/runs", headers={"Authorization": "Bearer " + self.token})
                    response = conn.getresponse()
                    self.assertEqual((response.status, json.load(response)), (200, {"run_ids": [self.report["run_id"]]}))
            finally:
                http.shutdown()
                thread.join()
