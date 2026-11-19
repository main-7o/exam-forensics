import itertools
import json
import math
import os
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

import numpy as np

from exam_forensics import ai_collect
from exam_forensics.__main__ import main
from exam_forensics.core import _gamma_pq, analyze, chi2_two_sided, copy_lr_p, sum_tail, tail_probs
from exam_forensics.simulate import simulate


def run(sim, **kw):
    return analyze(sim["responses"], sim["key"], sim["ai_runs"], sim["groups"], sim["prior"],
                   sim["times"], sim["leaked"], **kw)


class ExactDistributions(unittest.TestCase):
    def test_tail_matches_binomial(self):
        m, p = 10, 0.3
        tail = tail_probs(np.full(m, p))
        for x in range(m + 1):
            want = sum(math.comb(m, k) * p**k * (1 - p) ** (m - k) for k in range(x, m + 1))
            self.assertAlmostEqual(tail[x], want, places=12)

    def test_sum_tail_matches_brute_force(self):
        rng = np.random.default_rng(0)
        probs = rng.dirichlet(np.ones(3), size=(1, 4))
        values = np.round(rng.normal(0, 1, (1, 4, 3)) / 0.02) * 0.02  # 격자 위 값이라 반올림 오차 없음
        tail, iv = sum_tail(values, probs)
        for combo in itertools.product(range(3), repeat=4):
            idx = sum(iv[0, j, k] for j, k in enumerate(combo))
            s = sum(values[0, j, k] for j, k in enumerate(combo))
            want = sum(np.prod([probs[0, j, k] for j, k in enumerate(c)])
                       for c in itertools.product(range(3), repeat=4)
                       if sum(values[0, j, k] for j, k in enumerate(c)) >= s - 1e-9)
            self.assertAlmostEqual(tail[0, idx], want, places=10)

    def test_chi2(self):
        lower, upper = _gamma_pq(1.0, 2.0)  # df=2, x=4: 상측 = e^-2
        self.assertAlmostEqual(upper, math.exp(-2), places=12)
        self.assertAlmostEqual(chi2_two_sided(4.0, 2), 2 * math.exp(-2), places=12)

    def test_copy_lr_identical_is_extreme(self):
        probs = np.full((20, 6), 1 / 6)
        a = np.arange(20) % 5 + 1
        self.assertLess(copy_lr_p(a, a, probs), 1e-10)
        self.assertEqual(copy_lr_p(np.zeros(20, dtype=int), a, probs), 1.0)


class EndToEnd(unittest.TestCase):
    def test_detects_heavy_cheating(self):
        sim = simulate(n=3000, copy_rate=0.02, ai_rate=0.02, leak_rate=0.02, copy_frac=1.0, ai_frac=1.0,
                       n_leaked=20, leak_known=1.0, seed=1)
        r, t = run(sim), sim["truth"]
        f = r["any_flag"]
        self.assertGreater(f[t == "copier"].mean(), 0.5)
        self.assertGreater(f[t == "ai"].mean(), 0.5)
        self.assertGreater(f[t == "leak"].mean(), 0.5)
        self.assertLessEqual(f[t == "honest"].sum(), 2)
        self.assertGreater((r["cluster"] >= 0).sum(), 0)

    def test_no_cheaters_no_flags(self):
        sim = simulate(n=3000, copy_rate=0, ai_rate=0, leak_rate=0, seed=2)
        r = run(sim)
        self.assertLessEqual(r["any_flag"].sum(), 1)
        self.assertLessEqual(r["room_flag"].sum(), 1)

    def test_rejects_bad_input(self):
        with self.assertRaises(ValueError):
            analyze(np.ones((5, 3), dtype=int), np.ones(4, dtype=int))
        with self.assertRaises(ValueError):
            analyze(np.ones((5, 3), dtype=int), np.ones(3, dtype=int), times=np.ones((5, 2)))


class Robustness(unittest.TestCase):
    def test_poisson_sf(self):
        from exam_forensics.core import poisson_sf
        lam, x = 2.5, 4
        want = 1 - sum(math.exp(-lam) * lam**k / math.factorial(k) for k in range(x))
        self.assertAlmostEqual(float(poisson_sf(np.array([x]), np.array([lam]))[0]), want, places=12)

    def test_school_adjustment_removes_false_alarms(self):
        sim = simulate(n=3000, copy_rate=0, ai_rate=0, leak_rate=0, academy_by_room=True,
                       academy_sd=0.8, academy_conc=3, seed=3)
        plain = analyze(sim["responses"], sim["key"], groups=sim["groups"])
        fixed = analyze(sim["responses"], sim["key"], groups=sim["groups"], school=sim["academy"])
        self.assertGreater(plain["flags"]["답안공유"].sum(), 5)  # 보정 없으면 학교 효과를 베끼기로 오해
        self.assertTrue(plain["warnings"])
        self.assertLessEqual(fixed["flags"]["답안공유"].sum(), 1)

    def test_cross_room_lsh_finds_phone_sharing(self):
        sim = simulate(n=3000, copy_rate=0.01, ai_rate=0, leak_rate=0, copy_frac=1.0, copy_anywhere=True, seed=4)
        t = sim["truth"]
        r = analyze(sim["responses"], sim["key"], groups=sim["groups"], cross_room=True)
        self.assertGreater(r["flags"]["답안공유"][t == "copier"].mean(), 0.5)
        self.assertLessEqual(r["flags"]["답안공유"][t == "honest"].sum(), 1)
        self.assertTrue(any(p[7] == "전체" and p[6] for p in r["pairs"]))


class Cli(unittest.TestCase):
    def test_simulate_then_analyze(self):
        with tempfile.TemporaryDirectory() as d:
            main(["simulate", "--out", d, "--n", "600"])
            with open(os.path.join(d, "leaked.txt"), encoding="utf-8") as f:
                leaked = f.read()
            out = os.path.join(d, "report")
            main(["analyze", os.path.join(d, "responses.csv"), "--key", os.path.join(d, "key.csv"),
                  "--ai", os.path.join(d, "ai.csv"), "--times", os.path.join(d, "times.csv"),
                  "--leaked", leaked, "--out", out])
            with open(os.path.join(out, "examinees.csv"), encoding="utf-8-sig") as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 601)
            self.assertIn("사전유출p", lines[0])
            self.assertIn("응답시간p", lines[0])
            for name in ["pairs.csv", "rooms.csv"]:
                self.assertTrue(os.path.exists(os.path.join(out, name)))


class FakeChat(BaseHTTPRequestHandler):
    """OpenAI 호환 가짜 서버: 항상 '정답: 3'."""
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        assert body["messages"][0]["role"] == "user"
        out = json.dumps({"choices": [{"message": {"content": "풀이...\n정답: 3"}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *a):
        pass


class Collector(unittest.TestCase):
    def test_parse_answer(self):
        self.assertEqual(ai_collect.parse_answer("정답: 2\n아니 정답: 4", 5), 4)
        self.assertEqual(ai_collect.parse_answer("정답：(5)", 5), 5)
        self.assertEqual(ai_collect.parse_answer("정답: 7", 5), 0)
        self.assertEqual(ai_collect.parse_answer("모르겠다", 5), 0)

    def test_collect_with_fake_server(self):
        server = HTTPServer(("127.0.0.1", 0), FakeChat)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            qs = [{"id": f"q{i}", "question": "1+1?", "choices": ["1", "2", "3", "4", "5"]} for i in (1, 2)]
            url = f"http://127.0.0.1:{server.server_port}/v1"
            results, at = ai_collect.collect(qs, "openai", "fake", runs=2, base_url=url, log=lambda *_: None)
            self.assertEqual(results, [{"q1": 3, "q2": 3}] * 2)
            with tempfile.TemporaryDirectory() as d:
                path = os.path.join(d, "ai.csv")
                ai_collect.append_csv(path, "fake", results, at, ["q1", "q2"])
                ai_collect.append_csv(path, "fake2", results, at, ["q1", "q2"])
                with open(path, encoding="utf-8-sig") as f:
                    text = f.read()
                self.assertEqual(text.count(chr(0xFEFF)), 0)  # 이어쓰기에서 BOM이 끼지 않음
                self.assertEqual(len(text.strip().splitlines()), 5)
        finally:
            server.shutdown()
            server.server_close()


if __name__ == "__main__":
    unittest.main()
