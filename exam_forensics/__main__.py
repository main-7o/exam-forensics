"""명령줄 도구. 사용법은 README.md 참고.

python -m exam_forensics analyze responses.csv --key key.csv [--ai ai.csv] [--times times.csv] [--leaked q3,q7]
python -m exam_forensics collect-ai questions.json --provider anthropic --model claude-opus-5-5 --out ai.csv
python -m exam_forensics simulate --out sample
python -m exam_forensics demo
"""
import argparse
import csv
import os
import sys

import numpy as np

from .core import analyze
from .simulate import simulate

META = {"id", "group", "school", "prior", "model", "run", "collected_at"}
TRUTH_KINDS = ["ai", "leak", "copier", "source", "honest"]


def read_table(path):
    with open(path, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"{path}: 데이터가 없습니다")
    return rows


def to_int(v, where):
    v = (v or "").strip()
    if v == "":
        return 0
    try:
        return int(v)
    except ValueError:
        raise ValueError(f"{where}: 정수가 아닌 값 '{v}'") from None


def to_float(v):
    v = (v or "").strip()
    return float(v) if v else np.nan


def load(args):
    rows = read_table(args.responses)
    items = [c for c in rows[0].keys() if c not in META]
    resp = np.array([[to_int(r[c], f"{args.responses} id={r.get('id')} {c}") for c in items] for r in rows])
    ids = [r.get("id") or str(i) for i, r in enumerate(rows)]
    data = {"ids": ids, "items": items, "responses": resp, "groups": None, "school": None, "prior": None,
            "ai_runs": None, "times": None, "leaked": None}
    for col, name in [("group", "groups"), ("school", "school")]:
        if col in rows[0]:
            data[name] = np.array([r[col] for r in rows])
    if "prior" in rows[0]:
        vals = [r["prior"].strip() for r in rows]
        if all(vals):
            data["prior"] = np.array([float(v) for v in vals])
        else:
            print("경고: prior 열에 빈 값이 있어 성적 급상승 분석을 건너뜁니다", file=sys.stderr)

    key_row = read_table(args.key)[0]
    missing = [c for c in items if c not in key_row]
    if missing:
        raise ValueError(f"정답 파일에 문항이 없습니다: {missing[:5]}")
    data["key"] = np.array([to_int(key_row[c], f"{args.key} {c}") for c in items])

    if args.ai:
        runs = {}
        for i, r in enumerate(read_table(args.ai)):
            runs.setdefault(r.get("model") or "ai", []).append(
                [to_int(r.get(c), f"{args.ai} {c}") for c in items])
        data["ai_runs"] = {k: np.array(v) for k, v in runs.items()}

    if args.times:
        by_id = {r.get("id"): r for r in read_table(args.times)}
        lost = [i for i in ids if i not in by_id]
        if lost:
            raise ValueError(f"응답시간 파일에 없는 수험생: {lost[:5]}")
        data["times"] = np.array([[to_float(by_id[i].get(c)) for c in items] for i in ids])

    if args.leaked:
        names = [s.strip() for s in args.leaked.split(",") if s.strip()]
        bad = [s for s in names if s not in items]
        if bad:
            raise ValueError(f"없는 문항: {bad}")
        data["leaked"] = [items.index(s) for s in names]
    return data


def write_csv(path, header, rows):
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def fmt(v):
    return f"{v:.3e}"


def report(res, data, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    ids = data["ids"]
    cluster = res["cluster"]

    write_csv(os.path.join(out_dir, "pairs.csv"),
              ["id_a", "id_b", "구분", "같은오답수", "기대값", "같은오답p", "우도비p", "판정", "공모집단"],
              [[ids[a], ids[b], fam, x, f"{e:.2f}", fmt(pc), fmt(pl), "Y" if f else "",
                cluster[a] if f else ""]
               for a, b, x, e, pc, pl, f, fam in sorted(res["pairs"], key=lambda t: min(t[4], t[5]))])

    header = ["id", "점수", "점수대", "거트만G*", "이상응답p"]
    cols = [ids, res["scores"], res["bands"], [f"{g:.3f}" for g in res["gstar"]],
            [fmt(v) for v in res["p"]["이상응답"]]]
    if "ai" in res:
        ai = res["ai"]
        for t, name in enumerate(ai["models"]):
            header += [f"{name}_우도비p", f"{name}_AI오답일치", f"{name}_기대값", f"{name}_오답p"]
            cols += [[fmt(v) for v in ai["lr_p"][:, t]], ai["wrong_obs"][:, t],
                     [f"{v:.2f}" for v in ai["wrong_exp"][:, t]], [fmt(v) for v in ai["wrong_p"][:, t]]]
    if "preknowledge" in res:
        obs, p = res["preknowledge"]
        header += ["유출문항정답수", "사전유출p"]
        cols += [obs, [fmt(v) for v in p]]
    if "응답시간" in res["p"]:
        header += ["응답시간p", "실력대비속도z(참고)"]
        cols += [[fmt(v) for v in res["p"]["응답시간"]], [f"{v:.2f}" for v in res["speed_z"]]]
    if "gain" in res:
        z, p = res["gain"]
        header += ["급상승z", "급상승p"]
        cols += [[f"{v:.2f}" for v in z], [fmt(v) for v in p]]
    header += ["공모집단", "판정근거", "조사대상"]
    reasons = [", ".join(k for k, f in res["flags"].items() if f[i]) for i in range(len(ids))]
    cols += [[c if c >= 0 else "" for c in cluster], reasons, ["Y" if f else "" for f in res["any_flag"]]]
    write_csv(os.path.join(out_dir, "examinees.csv"), header, list(zip(*cols)))

    if "rooms" in res:
        names = np.unique(data["groups"])
        pairs, hits, p = res["rooms"]
        write_csv(os.path.join(out_dir, "rooms.csv"), ["group", "쌍수", "유사쌍수", "기대값", "p값", "판정"],
                  [[g, int(t), int(h), f"{t * 1e-3:.2f}", fmt(pv), "Y" if f else ""]
                   for g, t, h, pv, f in zip(names, pairs, hits, p, res["room_flag"])])


def save_sample(sim, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    m = sim["key"].shape[0]
    items = [f"q{j + 1}" for j in range(m)]
    write_csv(os.path.join(out_dir, "responses.csv"), ["id", "group", "prior"] + items,
              [[f"s{i}", g, f"{p:.3f}"] + list(r)
               for i, (g, p, r) in enumerate(zip(sim["groups"], sim["prior"], sim["responses"]))])
    write_csv(os.path.join(out_dir, "times.csv"), ["id"] + items,
              [[f"s{i}"] + [f"{t:.1f}" for t in row] for i, row in enumerate(sim["times"])])
    write_csv(os.path.join(out_dir, "key.csv"), items, [list(sim["key"])])
    write_csv(os.path.join(out_dir, "ai.csv"), ["model", "run", "collected_at"] + items,
              [[name, i + 1, "simulated"] + list(r)
               for name, runs in sim["ai_runs"].items() for i, r in enumerate(runs)])
    write_csv(os.path.join(out_dir, "truth.csv"), ["id", "truth"],
              [[f"s{i}", t] for i, t in enumerate(sim["truth"])])
    with open(os.path.join(out_dir, "leaked.txt"), "w", encoding="utf-8") as f:
        f.write(",".join(items[j] for j in sim["leaked"]))


def evaluate(truth, flagged):
    """유형별 탐지율과 정직한 수험생 오탐 수."""
    flagged = np.asarray(flagged)
    cells = []
    for kind in TRUTH_KINDS:
        mask = truth == kind
        if mask.any():
            cells.append(f"{kind} {flagged[mask].sum()}/{mask.sum()}")
    return "  ".join(cells)


def run_demo(n, seed):
    sim = simulate(n=n, seed=seed)
    res = analyze(sim["responses"], sim["key"], sim["ai_runs"], sim["groups"], sim["prior"],
                  sim["times"], sim["leaked"])
    print(f"수험생 {n}명, 유형별 1%씩 부정행위 (seed {seed})\n")
    for name, f in res["flags"].items():
        print(f"[{name:>4}] {evaluate(sim['truth'], f)}")
    print(f"[ 전체] {evaluate(sim['truth'], res['any_flag'])}")
    if "room_flag" in res:
        print(f"\n고사실 판정 {res['room_flag'].sum()}곳 / {len(res['room_flag'])}곳")


def run_collect(args):
    from .ai_collect import append_csv, collect, load_questions  # 네트워크 코드는 여기서만 불러온다
    qs = load_questions(args.questions)
    results, at = collect(qs, args.provider, args.model, args.runs, args.base_url, args.api_key_env)
    append_csv(args.out, args.label or args.model, results, at, [q["id"] for q in qs])
    print(f"{args.out} 에 {args.runs}회분 추가 (수집 시각 {at})")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="exam_forensics", description="로컬 시험 데이터 포렌식")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("analyze", help="응답 CSV 분석 (오프라인)")
    a.add_argument("responses")
    a.add_argument("--key", required=True)
    a.add_argument("--ai", help="AI 응답 CSV (model 열 + 문항 열, 같은 모델 여러 행 가능)")
    a.add_argument("--times", help="문항별 응답시간(초) CSV (id 열 + 문항 열)")
    a.add_argument("--leaked", help="유출 의심 문항, 쉼표로 구분 (예: q3,q7)")
    a.add_argument("--cross-room", action="store_true",
                   help="고사실을 넘는 답안 공유(휴대폰 등)도 LSH로 찾기 (group 열이 있을 때)")
    a.add_argument("--out", default="report")
    a.add_argument("--alpha", type=float, default=0.05, help="방법별 본페로니 보정 전 유의수준")

    c = sub.add_parser("collect-ai", help="AI에게 시험 문제를 풀게 해 ai.csv에 추가 (인터넷 사용)")
    c.add_argument("questions", help="문제 JSON")
    c.add_argument("--provider", choices=["anthropic", "openai"], required=True,
                   help="anthropic: 공식 SDK (ANTHROPIC_API_KEY). openai: OpenAI 호환 API (Gemini, Ollama 등)")
    c.add_argument("--model", required=True)
    c.add_argument("--base-url", help="openai 호환 주소 (예: https://api.openai.com/v1, http://localhost:11434/v1)")
    c.add_argument("--api-key-env", help="API 키가 든 환경변수 이름 (예: OPENAI_API_KEY)")
    c.add_argument("--runs", type=int, default=5)
    c.add_argument("--label", help="ai.csv에 기록할 모델 이름 (기본: --model)")
    c.add_argument("--out", default="ai.csv")

    s = sub.add_parser("simulate", help="정답을 아는 가상 데이터 생성")
    s.add_argument("--out", default="sample")
    s.add_argument("--n", type=int, default=10000)
    s.add_argument("--seed", type=int, default=0)

    d = sub.add_parser("demo", help="가상 데이터로 탐지율·오탐 측정")
    d.add_argument("--n", type=int, default=10000)
    d.add_argument("--seed", type=int, default=0)

    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        data = load(args)
        res = analyze(data["responses"], data["key"], data["ai_runs"], data["groups"], data["prior"],
                      data["times"], data["leaked"], school=data["school"], cross_room=args.cross_room,
                      alpha=args.alpha)
        report(res, data, args.out)
        print(f"{len(data['ids'])}명 분석, 조사대상 {int(res['any_flag'].sum())}명. 결과: {args.out}/")
        for w in res["warnings"]:
            print(f"경고: {w}")
        print("주의: 통계 신호는 조사 우선순위일 뿐 부정행위의 증거가 아닙니다.")
    elif args.cmd == "collect-ai":
        run_collect(args)
    elif args.cmd == "simulate":
        save_sample(simulate(n=args.n, seed=args.seed), args.out)
        print(f"{args.out}/ 에 responses.csv, times.csv, key.csv, ai.csv, truth.csv, leaked.txt 생성")
    else:
        run_demo(args.n, args.seed)


if __name__ == "__main__":
    main()
