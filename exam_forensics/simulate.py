"""정답(누가 부정행위자인지)을 아는 가상 시험 데이터 생성기. 탐지 성능 검증용."""
import numpy as np


def simulate(n=10000, m=45, n_options=5, room_size=28, copy_rate=0.01, ai_rate=0.01, leak_rate=0.01,
             copy_frac=0.5, ai_frac=0.8, ai_runs=5, ai_consistency=0.85, n_leaked=8, leak_known=0.9,
             n_academies=0, academy_sd=0.0, academy_conc=0.0, academy_by_room=False, copy_anywhere=False,
             seed=0):
    """3PL 문항반응 모형 + 문항별 오답 매력도 + 로그정규 응답시간.

    부정행위 유형 (truth 값)
    - copier/source: 같은 고사실 원본(source)의 답을 copy_frac 비율 문항에서 베낌
    - ai: 자기가 돌린 AI 답을 ai_frac 비율 문항에 사용. 답을 받아 적느라 응답시간이 일정해짐
    - leak: 유출된 n_leaked 문항의 답을 미리 알고 leak_known 확률로 맞히며 빨리 답함
    조사자는 같은 AI를 ai_runs번 돌린 결과를 갖고 있다 (AI 출력은 실행마다 조금씩 다름).
    """
    rng = np.random.default_rng(seed)
    theta = rng.normal(0, 1, n)
    a = rng.uniform(0.8, 2.0, m)
    b = rng.normal(0, 1, m)
    c = 1.0 / n_options
    key = rng.integers(1, n_options + 1, m)
    attract = rng.dirichlet(np.full(n_options - 1, 1.0), m)

    def answer(p_correct, weights, rows):
        out = np.empty((rows, m), dtype=np.int64)
        right = rng.random((rows, m)) < p_correct
        for j in range(m):
            distractors = np.array([k for k in range(1, n_options + 1) if k != key[j]])
            picks = rng.choice(distractors, size=rows, p=weights[j])
            out[:, j] = np.where(right[:, j], key[j], picks)
        return out

    # 학원 효과(모형 오설정 검증용): 같은 학원 학생은 문항별 난이도와 오답 습관을 공유한다.
    # 고사실과 무관하게 배정되므로 귀무 모형이 모르는 수험생 간 상관이 생긴다.
    # academy_by_room: 같은 학교 학생이 같은 고사실에 앉는 경우 (가장 가혹한 조건)
    if academy_by_room:
        academy = np.arange(n) // room_size
        n_academies = int(academy.max()) + 1
    else:
        academy = rng.integers(0, max(n_academies, 1), n)
    responses = np.empty((n, m), dtype=np.int64)
    for ac in range(max(n_academies, 1)):
        rows = np.nonzero(academy == ac)[0]
        b_ac = b + (rng.normal(0, academy_sd, m) if academy_sd > 0 else 0)
        w_ac = (np.array([rng.dirichlet(attract[j] * academy_conc) for j in range(m)])
                if academy_conc > 0 else attract)
        p = c + (1 - c) / (1 + np.exp(-1.7 * a * (theta[rows, None] - b_ac)))
        responses[rows] = answer(p, w_ac, len(rows))
    groups = np.arange(n) // room_size
    prior = theta + rng.normal(0, 0.5, n)

    # 응답시간: log t = beta_j - tau_i + e,  e ~ N(0, 1/alpha_j^2)
    beta = rng.normal(4.0, 0.4, m)          # 문항별 시간 강도 (약 55초)
    tau = rng.normal(0, 0.3, n)              # 수험생별 속도
    sd = 1 / rng.uniform(1.5, 2.5, m)
    times = np.exp(beta[None, :] - tau[:, None] + rng.normal(0, 1, (n, m)) * sd[None, :])
    truth = np.array(["honest"] * n, dtype=object)

    # AI: 사람과 다른 난이도 감각. 오답은 사람 매력도와 균등의 혼합.
    # 가정: AI는 문항마다 '선호 답'이 있고 실행마다 ai_consistency 확률로 그 답을 낸다
    ai_p = 1 / (1 + np.exp(-(2.0 - 0.5 * b + rng.normal(0, 1, m))))
    ai_weights = 0.5 * attract + 0.5 / (n_options - 1)
    preferred = answer(ai_p[None, :], ai_weights, 1)[0]

    def ai_run(rows):
        fresh = answer(ai_p[None, :], ai_weights, rows)
        keep = rng.random((rows, m)) < ai_consistency
        return np.where(keep, preferred[None, :], fresh)

    investigator = ai_run(ai_runs)

    order = rng.permutation(n)
    n_ai, n_leak, n_copy = int(n * ai_rate), int(n * leak_rate), int(n * copy_rate)
    for i in order[:n_ai]:
        own = ai_run(1)[0]
        use = rng.random(m) < ai_frac
        responses[i, use] = own[use]
        times[i, use] = rng.normal(60, 5, use.sum()).clip(20)  # 받아 적는 시간은 문항과 무관하게 일정
        truth[i] = "ai"

    leaked = np.sort(rng.choice(m, n_leaked, replace=False))
    for i in order[n_ai:n_ai + n_leak]:
        know = leaked[rng.random(n_leaked) < leak_known]
        responses[i, know] = key[know]
        times[i, know] *= 0.3
        truth[i] = "leak"

    for i in order[n_ai + n_leak:n_ai + n_leak + n_copy]:
        if truth[i] != "honest":
            continue
        same = np.ones(n, dtype=bool) if copy_anywhere else groups == groups[i]  # 휴대폰 공유는 고사실 무관
        mates = np.nonzero(same & (truth == "honest"))[0]
        mates = mates[mates != i]
        if len(mates) == 0:
            continue
        src = rng.choice(mates)
        use = rng.random(m) < copy_frac
        responses[i, use] = responses[src, use]
        truth[i], truth[src] = "copier", "source"

    return {"responses": responses, "key": key, "groups": groups, "prior": prior, "times": times,
            "ai_runs": {"ai": investigator}, "leaked": list(leaked), "truth": truth, "academy": academy}
