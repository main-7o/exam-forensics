"""실제 응답 위에 가짜 부정행위를 심어 탐지율과 오탐을 잰다 (Cizek & Wollack 데이터, 업체 비표시자만 사용).

실행: python benchmarks/plant_real.py <데이터 폴더>
실제 데이터의 상관 구조(학습 이력, 문항 간 의존성 등)를 그대로 둔 채 귀무 모형의 오탐을 확인한다.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import cizek_wollack as cw  # noqa: E402
from exam_forensics.core import analyze  # noqa: E402


def plant(r, key, rng, copy_rate, copy_frac, ai_rate, ai_frac, ai_acc=0.85, consistency=0.85, runs=5):
    n, m = r.shape
    r = r.copy()
    truth = np.array(["honest"] * n, dtype=object)
    preferred = np.where(rng.random(m) < ai_acc, key, (key + rng.integers(1, 4, m) - 1) % 4 + 1)

    def ai_run(k):
        fresh = np.where(rng.random((k, m)) < ai_acc, key, (key + rng.integers(1, 4, (k, m)) - 1) % 4 + 1)
        return np.where(rng.random((k, m)) < consistency, preferred, fresh)

    order = rng.permutation(n)
    n_ai, n_copy = int(n * ai_rate), int(n * copy_rate)
    for i in order[:n_ai]:
        use = rng.random(m) < ai_frac
        r[i, use] = ai_run(1)[0][use]
        truth[i] = "ai"
    for i in order[n_ai:n_ai + n_copy]:
        if truth[i] != "honest":
            continue
        src = rng.choice(np.nonzero(truth == "honest")[0])
        if src == i:
            continue
        use = rng.random(m) < copy_frac
        r[i, use] = r[src, use]
        truth[i], truth[src] = "copier", "source"
    return r, truth, {"ai": ai_run(runs)}


def main(folder):
    resp, administered, times, flagged = cw.load(folder)
    pats, counts = np.unique(administered, axis=0, return_counts=True)
    print("실제 응답(업체 비표시자) 위 심기 실험. 각 조건 seed 3개 합계\n")
    for label, kw in [("부정 없음", dict(copy_rate=0, copy_frac=0, ai_rate=0, ai_frac=0)),
                      ("베낌 70% / AI 80%", dict(copy_rate=.01, copy_frac=.7, ai_rate=.01, ai_frac=.8)),
                      ("베낌 100% / AI 100%", dict(copy_rate=.01, copy_frac=1., ai_rate=.01, ai_frac=1.))]:
        tot = {"copier": [0, 0], "ai": [0, 0], "fp": 0, "honest": 0}
        for f in range(2):
            pat = pats[np.argsort(-counts)[f]]
            idx = np.nonzero((administered == pat).all(axis=1) & ~flagged)[0]
            items = np.nonzero(pat)[0]
            base = resp[np.ix_(idx, items)]
            key = cw.infer_key(base)
            for seed in range(3):
                r, truth, ai_runs = plant(base, key, np.random.default_rng(seed), **kw)
                res = analyze(r, key, ai_runs=ai_runs)
                fl = res["any_flag"]
                for k in ["copier", "ai"]:
                    tot[k][0] += int(fl[truth == k].sum())
                    tot[k][1] += int((truth == k).sum())
                tot["fp"] += int(fl[truth == "honest"].sum())
                tot["honest"] += int((truth == "honest").sum())
        det = "  ".join(f"{k} {a}/{b}" for k, (a, b) in [("베낀사람", tot["copier"]), ("AI", tot["ai"])] if b)
        print(f"[{label}] {det}  정직 오탐 {tot['fp']}/{tot['honest']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
