"""실제 자격시험 데이터 검증 (Cizek & Wollack, 2017, Handbook of Quantitative Methods for Detecting Cheating on Tests).

데이터는 저장소에 포함하지 않는다. 받는 곳 (핸드북을 출처로 밝혀 달라는 조건):
  https://cengiz.me/posts/xgboost/data/train.csv
  https://cengiz.me/posts/xgboost/data/test.csv

실행: python benchmarks/cizek_wollack.py <train.csv와 test.csv가 있는 폴더>

라벨(Flagged)은 시험 업체가 '문항에 미리 접근했다고 의심한' 응시자다. 확정된 정답이 아니다.
정답 키가 데이터에 없어서 나머지 점수 기준 반복 추정으로 구한다.
"""
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from exam_forensics.core import analyze  # noqa: E402

N_ITEMS, N_OPT = 253, 4


def load(folder):
    rows = []
    for name in ["train.csv", "test.csv"]:
        with open(os.path.join(folder, name), encoding="utf-8") as f:
            r = csv.reader(f)
            header = next(r)
            rows += list(r)
    v = np.array([[x[i] for i in range(N_ITEMS * N_OPT)] for x in rows])
    onehot = np.where(v == "NA", "-1", v).astype(int).reshape(len(rows), N_ITEMS, N_OPT)
    administered = (onehot >= 0).all(axis=2)
    chosen = onehot.clip(0)
    resp = np.where(administered & (chosen.sum(axis=2) > 0), chosen.argmax(axis=2) + 1, 0)
    t_cols = [i for i, c in enumerate(header) if c.startswith("idur.")]
    times = np.array([[float(x[i]) if x[i] not in ("NA", "") else np.nan for i in t_cols] for x in rows])
    flagged = np.array([x[-1] == "1" for x in rows])
    return resp, administered, times, flagged


def infer_key(resp, n_iter=10):
    """정답 추정: 그 선택지를 고른 사람들의 나머지 점수 평균이 가장 높은 선택지."""
    key = np.array([np.bincount(resp[:, j][resp[:, j] > 0], minlength=N_OPT + 1)[1:].argmax() + 1
                    for j in range(resp.shape[1])])
    for _ in range(n_iter):
        correct = resp == key
        total = correct.sum(axis=1)
        new = key.copy()
        for j in range(resp.shape[1]):
            rest = total - correct[:, j]
            means = [rest[resp[:, j] == k].mean() if (resp[:, j] == k).sum() >= 20 else -1
                     for k in range(1, N_OPT + 1)]
            new[j] = int(np.argmax(means)) + 1
        if (new == key).all():
            break
        key = new
    return key


def avg_rank(score):
    """1부터 시작하는 순위, 동점은 평균 순위 (데이터 순서가 결과에 끼어들지 않게)."""
    order = np.argsort(score, kind="stable")
    s = score[order]
    ranks = np.empty(len(score))
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return ranks


def auc(score, label):
    """점수가 클수록 의심. 순위 기반 AUC (동점은 평균 순위)."""
    ranks = avg_rank(score)
    pos = label.sum()
    return (ranks[label].sum() - pos * (pos + 1) / 2) / (pos * (len(label) - pos))


def power_at_fpr(score, label, fpr):
    thr = np.quantile(score[~label], 1 - fpr)
    return (score[label] > thr).mean()


def suspicion_scores(res, n):
    """방법별 의심 점수 = -log10 p (쌍 검정은 그 사람이 낀 쌍 중 최소 p)."""
    pair_p = np.ones(n)
    for a, b, x, e, pc, pl, f, fam in res["pairs"]:
        p = min(pc, pl)
        pair_p[a] = min(pair_p[a], p)
        pair_p[b] = min(pair_p[b], p)
    scores = {"답안공유": pair_p, "이상응답": res["p"]["이상응답"]}
    if "응답시간" in res["p"]:
        scores["응답시간"] = res["p"]["응답시간"]
    out = {k: -np.log10(np.clip(v, 1e-300, 1)) for k, v in scores.items()}
    if "speed_z" in res:
        out["실력대비속도"] = res["speed_z"]  # 판정 없는 순위 점수
    return out


def main(folder):
    resp, administered, times, flagged = load(folder)
    forms = {}
    for i, pattern in enumerate(map(tuple, administered)):
        forms.setdefault(pattern, []).append(i)
    small = sum(len(v) for v in forms.values() if len(v) < 100)
    print(f"응시자 {len(resp)}명, 업체 표시 {flagged.sum()}명, 시험지 {len(forms)}종 "
          f"(응시 문항 조합이 특이한 {small}명은 시험지당 100명 미만이라 제외)\n")

    all_scores, all_flag, all_label = {}, [], []
    for f_id, (pattern, idx) in enumerate(sorted(forms.items(), key=lambda kv: -len(kv[1]))):
        idx = np.array(idx)
        items = np.nonzero(np.array(pattern))[0]
        if len(idx) < 100:
            continue
        r = resp[np.ix_(idx, items)]
        t = times[np.ix_(idx, items)]
        key = infer_key(r)
        p_values = (r == key).mean(axis=0)
        res = analyze(r, key, times=t)
        lab = flagged[idx]
        sc = suspicion_scores(res, len(idx))
        print(f"[시험지 {f_id}] {len(idx)}명 × {len(items)}문항, 업체 표시 {lab.sum()}명, "
              f"추정 정답률 평균 {p_values.mean():.2f}, 정답률 0.25 미만 문항 {(p_values < 0.25).sum()}개")
        for name, s in sc.items():
            fl = res["flags"].get(name)
            judged = f"판정 {fl.sum()}명 중 업체 표시 {(fl & lab).sum()}명" if fl is not None else "판정 없음(순위 점수)"
            print(f"   {name:>6}: AUC {auc(s, lab):.3f} | {judged} "
                  f"| 오탐 1%/5%일 때 탐지 {power_at_fpr(s, lab, .01):.0%}/{power_at_fpr(s, lab, .05):.0%}")
        any_f = res["any_flag"]
        print(f"   전체 판정 {any_f.sum()}명 (업체 표시 {(any_f & lab).sum()}명, 비표시 {(any_f & ~lab).sum()}명)\n")
        for name, s in sc.items():
            all_scores.setdefault(name, []).append(s)
        all_flag.append(any_f)
        all_label.append(lab)

    lab = np.concatenate(all_label)
    print("[두 시험지 합산]")
    ranks = []
    for name, parts in all_scores.items():
        s = np.concatenate(parts)
        # 시험지 안에서 백분위 순위로 바꿔 척도를 맞춘다
        ranks.append(np.concatenate([(avg_rank(p) - 1) / (len(p) - 1) for p in parts]))
        print(f"   {name:>6}: AUC {auc(s, lab):.3f} | 오탐 1%/5%일 때 탐지 "
              f"{power_at_fpr(s, lab, .01):.0%}/{power_at_fpr(s, lab, .05):.0%}")
    for label, combined in [("순위 최댓값 결합", np.max(ranks, axis=0)), ("순위 평균 결합", np.mean(ranks, axis=0))]:
        print(f"   {label}: AUC {auc(combined, lab):.3f} | 오탐 1%/5%일 때 탐지 "
              f"{power_at_fpr(combined, lab, .01):.0%}/{power_at_fpr(combined, lab, .05):.0%}")
    f = np.concatenate(all_flag)
    print(f"   본페로니 판정: {f.sum()}명 (업체 표시 {(f & lab).sum()}/{lab.sum()}명, 비표시 {(f & ~lab).sum()}명)")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
