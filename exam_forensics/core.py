"""시험 데이터 포렌식 핵심 통계. 모든 계산은 로컬에서만 수행한다 (네트워크 접근 없음).

입력 형식
- responses: (n, m) 정수 배열. 0 = 무응답, 1..K = 선택지
- key: (m,) 정답
- 귀무 모형: 같은 점수대(총점 분위) 수험생들은 문항별로 서로 독립적으로 응답하며,
  선택지 확률은 그 점수대의 실제 선택 비율이다 (비모수 명목 응답 모형).

모든 p값은 귀무 모형 아래에서 정확(exact)하게 계산한다. 연속 통계량은 GRID 간격으로
이산화한 뒤 그 이산 통계량의 분포를 동적계획법으로 정확히 구하므로, 근사 없이 유효한 검정이다.
"""
import math

import numpy as np

SMOOTH = 0.5    # 라플라스 평활: 점수대가 작을 때 확률 0을 막는다
PRELIM = 1e-3   # 2단계 추정에서 확률 추정에서 뺄 1차 의심 기준 (판정 기준 아님)
GRID = 0.02     # 우도비 통계량 이산화 간격
ROOM_P = 1e-3   # 고사실 분석: 이 p값 미만인 쌍의 개수를 센다
F_COPY = 0.5    # 답안 공유 대립가설: 베낀 문항 비율
F_AI = 0.7      # AI 대립가설: AI 답을 쓴 문항 비율
SCHOOL_KAPPA = 2.0      # 학교 보정 축소 강도. 강한 학교효과 가상실험에서 0.5/2/5/15 중 오탐 0을 지키는 최대 탐지력
ALL_PAIRS_MAX = 5e7     # 그룹 없이 모든 쌍을 정확히 훑는 최대 쌍 수. 넘으면 LSH
ROOM_WARN = 0.05        # 판정된 고사실 비율이 이보다 크면 구조적 상관 경고


# ---------------------------------------------------------------- 기본 도구

def score_bands(scores, n_bands=20):
    """총점 분위수로 점수대를 나눈다. 같은 점수는 항상 같은 점수대."""
    n_bands = max(1, min(n_bands, len(scores) // 50))
    edges = np.quantile(scores, np.linspace(0, 1, n_bands + 1)[1:-1])
    raw = np.searchsorted(edges, scores, side="right")
    _, bands = np.unique(raw, return_inverse=True)
    return bands


def option_probs(responses, bands, n_options, include=None):
    """prop[b, j, k] = 점수대 b 수험생이 문항 j에서 k(0=무응답)를 고를 확률.

    include: 추정에 쓸 수험생 마스크 (1차 의심자를 빼서 오염을 막는다)
    """
    n_bands = bands.max() + 1
    m = responses.shape[1]
    if include is None:
        include = np.ones(len(bands), dtype=bool)
    counts = np.zeros((n_bands, m, n_options + 1))
    for b in range(n_bands):
        sub = responses[(bands == b) & include]
        for k in range(n_options + 1):
            counts[b, :, k] = (sub == k).sum(axis=0)
    totals = counts.sum(axis=2, keepdims=True)
    return (counts + SMOOTH) / (totals + SMOOTH * (n_options + 1))


def tail_probs(p):
    """포아송-이항 분포 꼬리확률. p: (..., m) -> tail[..., x] = P(X >= x)."""
    m = p.shape[-1]
    dist = np.zeros(p.shape[:-1] + (m + 1,))
    dist[..., 0] = 1.0
    for j in range(m):
        pj = p[..., j : j + 1]
        shifted = np.zeros_like(dist)
        shifted[..., 1:] = dist[..., :-1] * pj
        dist = dist * (1 - pj) + shifted
    tail = np.cumsum(dist[..., ::-1], axis=-1)[..., ::-1]
    return np.clip(tail, 0.0, 1.0)


def sum_tail(values, probs):
    """S = sum_j values[b, j, k_j], k_j ~ probs[b, j, :] 독립일 때 점수대별 S의 정확한 꼬리확률.

    반환 (tail, iv): P(S >= s) = tail[b, idx], idx = sum_j iv[b, j, k_j]
    """
    iv = np.rint(values / GRID).astype(np.int64)
    iv -= iv.min(axis=2, keepdims=True)
    n_bands, m, n_k = iv.shape
    size = int(iv.max(axis=2).sum(axis=1).max()) + 1
    dist = np.zeros((n_bands, size))
    dist[:, 0] = 1.0
    for j in range(m):
        new = np.zeros_like(dist)
        for b in range(n_bands):
            for k in range(n_k):
                s = iv[b, j, k]
                new[b, s:] += dist[b, : size - s] * probs[b, j, k]
        dist = new
    tail = np.cumsum(dist[:, ::-1], axis=1)[:, ::-1]
    return np.clip(tail, 0.0, 1.0), iv


def _observed_idx(iv, bands, responses):
    m = responses.shape[1]
    return iv[bands[:, None], np.arange(m)[None, :], responses].sum(axis=1)


def binom_sf(k, n, p):
    """P(Bin(n, p) >= k)."""
    if k <= 0:
        return 1.0
    total = 0.0
    for x in range(k, n + 1):
        total += math.exp(math.lgamma(n + 1) - math.lgamma(x + 1) - math.lgamma(n - x + 1)
                          + x * math.log(p) + (n - x) * math.log1p(-p))
    return min(total, 1.0)


def chi2_two_sided(x, df):
    """카이제곱 양측 p값: 2 * min(하측, 상측). 하측은 '지나치게 일정함'을 잡는다."""
    lower, upper = _gamma_pq(df / 2.0, x / 2.0)
    return min(1.0, 2 * min(lower, upper))


def _gamma_pq(a, x):
    """정규화 불완전 감마 (P, Q). 작은 쪽을 직접 계산해 꼬리 정밀도를 지킨다."""
    if x <= 0:
        return 0.0, 1.0
    front = math.exp(-x + a * math.log(x) - math.lgamma(a))
    if x < a + 1:  # 급수로 P
        term = total = 1.0 / a
        ap = a
        for _ in range(1000):
            ap += 1
            term *= x / ap
            total += term
            if abs(term) < abs(total) * 1e-15:
                break
        p = min(1.0, total * front)
        return p, max(0.0, 1.0 - p)
    b, c, d = x + 1 - a, 1e300, 1 / (x + 1 - a)  # Lentz 연분수
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c
        c = c if abs(c) > 1e-300 else 1e-300
        h *= d * c
        if abs(d * c - 1) < 1e-15:
            break
    q = min(1.0, front * h)  # 연분수로 Q
    return max(0.0, 1.0 - q), q


# ---------------------------------------------------------------- 답안 공유

def _wrong_onehot(responses, key, n_options):
    """(n, m*K) 원-핫: 오답 선택지만 1. 무응답은 일치로 치지 않는다."""
    n, m = responses.shape
    wrong = (responses != key) & (responses > 0)
    onehot = np.zeros((n, m, n_options), dtype=np.float32)
    rows, cols = np.nonzero(wrong)
    onehot[rows, cols, responses[rows, cols] - 1] = 1
    return onehot.reshape(n, m * n_options)


def pair_screen(responses, key, bands, prop, groups=None, p_keep=1e-2, chunk=1000):
    """1단계: 같은 오답 일치 수(identical incorrect) 정확 검정으로 모든 쌍을 훑는다.

    groups(정수 배열)가 있으면 같은 그룹(고사실) 안의 쌍만 본다.
    반환: candidates [(a, b, 같은오답수, 기대값, p)], 검정한 쌍 수,
          그룹별 (쌍 수, p < ROOM_P 인 쌍 수)
    """
    n, m = responses.shape
    n_options = prop.shape[2] - 1
    wrong_prop = prop[:, :, 1:].copy()
    wrong_prop[:, np.arange(m), key - 1] = 0
    q = np.einsum("ajk,bjk->abj", wrong_prop, wrong_prop)
    tail = tail_probs(q)
    expected = q.sum(axis=2)

    w = _wrong_onehot(responses, key, n_options)
    n_groups = int(groups.max()) + 1 if groups is not None else 0
    room_pairs = np.zeros(n_groups, dtype=np.int64)
    room_hits = np.zeros(n_groups, dtype=np.int64)
    found, n_tests = [], 0
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x = (w[start:stop] @ w.T).astype(np.int64)
        eligible = np.arange(n)[None, :] > np.arange(start, stop)[:, None]
        if groups is not None:
            eligible &= groups[start:stop, None] == groups[None, :]
        n_tests += int(eligible.sum())
        pv = tail[bands[start:stop, None], bands[None, :], x]
        if groups is not None:
            g = groups[start:stop]
            np.add.at(room_pairs, g, eligible.sum(axis=1))
            np.add.at(room_hits, g, (eligible & (pv < ROOM_P)).sum(axis=1))
        for a, b in zip(*np.nonzero(eligible & (pv < p_keep))):
            ia = start + a
            found.append((ia, int(b), int(x[a, b]), float(expected[bands[ia], bands[b]]), float(pv[a, b])))
    return found, n_tests, (room_pairs, room_hits)


def copy_lr_p(src, cop, probs, f=F_COPY):
    """2단계: '이 사람(cop)이 src의 답을 f 비율로 베꼈다' 대 '독립' 우도비 검정의 정확 p값.

    일치한 문항마다 log(f / P(src의 답) + 1 - f)를 더한다. 드문 답(어려운 문항의 정답,
    인기 없는 오답)이 일치할수록 무겁게 센다. probs: cop 점수대의 (m, K+1) 선택 확률.
    """
    ok = src > 0
    pj = probs[ok, src[ok]]
    iw = np.rint(np.log(f / pj + (1 - f)) / GRID).astype(np.int64)
    size = int(iw.sum()) + 1
    dist = np.zeros(size)
    dist[0] = 1.0
    for wj, pjj in zip(iw, pj):
        new = dist * (1 - pjj)
        new[wj:] += dist[: size - wj] * pjj
        dist = new
    obs = int(iw[cop[ok] == src[ok]].sum())
    return float(min(1.0, dist[obs:].sum()))


def person_probs(bands, prop, school=None, responses=None, kappa=None):
    """수험생별 선택 확률을 돌려주는 함수 idx -> (len(idx), m, K+1).

    school이 있으면 학교(학원)별 선택 성향 w[s,j,k] = (관측 + kappa) / (기대 + kappa)로 기울인다.
    같이 배워서 생기는 공통 오답을 '베끼기'로 오해하지 않게 한다. kappa 축소 덕에 베낀 쌍
    하나가 학교 성향을 크게 바꾸지는 못한다.
    """
    if school is None:
        return lambda idx: prop[bands[idx]]
    kappa = SCHOOL_KAPPA if kappa is None else kappa
    n_school = int(school.max()) + 1
    k1 = prop.shape[2]
    w = np.ones((n_school, prop.shape[1], k1))
    for s in range(n_school):
        rows = np.nonzero(school == s)[0]
        if len(rows) == 0:
            continue
        obs = np.stack([(responses[rows] == k).sum(axis=0) for k in range(k1)], axis=1)
        exp = prop[bands[rows]].sum(axis=0)
        w[s] = (obs + kappa) / (exp + kappa)

    def probs(idx):
        p = prop[bands[idx]] * w[school[idx]]
        return p / p.sum(axis=2, keepdims=True)
    return probs


def explicit_pairs(responses, key, probs_of, ia, ib, p_keep, chunk=20000):
    """주어진 쌍 목록의 같은 오답 정확 검정 (수험생별 확률). 반환: 후보 리스트, 전체 p값."""
    m = responses.shape[1]
    out, all_p = [], np.ones(len(ia))
    for s in range(0, len(ia), chunk):
        a, b = ia[s:s + chunk], ib[s:s + chunk]
        pa, pb = probs_of(a), probs_of(b)
        pa[:, :, 0] = 0  # 무응답 일치는 세지 않는다
        pa[:, np.arange(m), key] = 0  # 정답 일치도 이 검정에서는 제외
        q = (pa * pb).sum(axis=2)
        ra, rb = responses[a], responses[b]
        x = ((ra == rb) & (ra != key) & (ra > 0)).sum(axis=1)
        e = q.sum(axis=1)
        # 사전 필터: 같은 평균의 포아송 꼬리는 포아송-이항 꼬리보다 크다. 그 값도 0.5 이상이면
        # 정확 계산할 필요가 없다 (어떤 판정 기준에도 못 미친다).
        sel = np.nonzero(poisson_sf(x, e) < 0.5)[0]
        pv = np.ones(len(a))
        if len(sel):
            pv[sel] = tail_probs(q[sel])[np.arange(len(sel)), x[sel]]
        all_p[s:s + chunk] = pv
        for t in np.nonzero(pv < p_keep)[0]:
            out.append((int(a[t]), int(b[t]), int(x[t]), float(e[t]), float(pv[t])))
    return out, all_p


def poisson_sf(x, lam):
    """P(Poisson(lam) >= x), 벡터."""
    x = np.asarray(x)
    term = np.exp(-lam)
    cdf = np.zeros_like(lam)
    for k in range(int(x.max()) if x.size else 0):
        cdf += np.where(k < x, term, 0.0)
        term = term * lam / (k + 1)
    return np.clip(1.0 - cdf, 0.0, 1.0)


def within_group_pairs(groups):
    """같은 그룹 안의 모든 쌍 (ia < ib 순서 아님, 쌍마다 한 번)."""
    order = np.argsort(groups, kind="stable")
    g = groups[order]
    starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
    ends = np.r_[starts[1:], len(g)]
    ia, ib = [], []
    for s, e in zip(starts, ends):
        i, j = np.triu_indices(e - s, 1)
        ia.append(order[s + i])
        ib.append(order[s + j])
    return np.concatenate(ia), np.concatenate(ib)


def lsh_candidates(responses, key, n_options, rows=None, n_lsh=40, min_wrong=4, max_bucket=200, seed=0,
                   chunk=256):
    """MinHash LSH: 오답 집합이 많이 겹치는 쌍만 후보로 뽑는다. 수십만 명도 O(n²) 없이 처리.

    rows(해시 묶음 크기)는 인원에 따라 자동으로 커진다: 1만 명 5, 10만 명 7, 55만 명 8.
    후보 수를 n에 거의 비례하게 묶는 대신, 부분 베끼기의 재현율이 떨어진다.
    한계: 근사 탐색이라 재현율이 100%가 아니다. 오답이 min_wrong개 미만인 고득점자는
    후보가 되지 않는다. 오탐은 늘지 않는다 (판정은 모든 쌍 수로 본페로니 보정).
    """
    n, m = responses.shape
    if rows is None:
        rows = int(np.clip(5 + round(2 * np.log10(max(n, 1) / 1e4)), 5, 9))
    wrong = (responses != key) & (responses > 0)
    n_tok = m * n_options
    tok = np.where(wrong, np.arange(m)[None, :] * n_options + responses - 1, n_tok)
    rng = np.random.default_rng(seed)
    prime = 2147483647
    h = rows * n_lsh
    hv = (rng.integers(1, prime, h)[:, None] * np.arange(n_tok)[None, :] + rng.integers(0, prime, h)[:, None]) % prime
    hv = np.concatenate([hv, np.full((h, 1), prime)], axis=1)  # 맞힌 문항 = 빈 토큰
    sig = np.empty((n, h), dtype=np.int64)
    for s in range(0, n, chunk):
        sig[s:s + chunk] = hv[:, tok[s:s + chunk]].min(axis=2).T
    ok = np.nonzero(wrong.sum(axis=1) >= min_wrong)[0]
    mult = rng.integers(1, 2 ** 31, rows)
    codes = []
    for t in range(n_lsh):
        keys = (sig[ok, t * rows:(t + 1) * rows] * mult).sum(axis=1)
        order = np.argsort(keys, kind="stable")
        k = keys[order]
        starts = np.flatnonzero(np.r_[True, k[1:] != k[:-1]])
        sizes = np.diff(np.r_[starts, len(k)])
        two = starts[sizes == 2]
        a, b = ok[order[two]], ok[order[two + 1]]
        codes.append(np.minimum(a, b) * n + np.maximum(a, b))
        big = (sizes > 2) & (sizes <= max_bucket)  # 너무 큰 버킷은 흔한 패턴이라 버린다
        for s, size in zip(starts[big], sizes[big]):
            mem = ok[order[s:s + size]]
            i, j = np.triu_indices(len(mem), 1)
            codes.append(np.minimum(mem[i], mem[j]) * n + np.maximum(mem[i], mem[j]))
    if not codes:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    code = np.unique(np.concatenate(codes))
    return code // n, code % n


def clusters(n, pairs):
    """판정된 쌍을 이어 공모 집단(연결 요소)을 만든다. 반환: 수험생별 집단 번호(-1 = 없음)."""
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in pairs:
        parent[find(a)] = find(b)
    members = {}
    for a, b in pairs:
        members.setdefault(find(a), set()).update((a, b))
    label = np.full(n, -1)
    for cid, group in enumerate(sorted(members.values(), key=lambda s: -len(s))):
        label[list(group)] = cid
    return label


# ---------------------------------------------------------------- 개인 지표

def person_fit(responses, bands, prop):
    """이상응답: 같은 점수대 기준으로 응답 패턴 전체가 얼마나 드문가 (정확 p값).

    통계량은 패턴의 놀라움(-log 확률)의 합. 문항반응이론 가정 없이 선택지 정보까지 쓴다.
    """
    tail, iv = sum_tail(-np.log(prop), prop)
    return tail[bands, _observed_idx(iv, bands, responses)]


def ai_distribution(ai_answers, n_options):
    """한 모델의 여러 실행 결과 (runs, m) -> 문항별 AI 선택 분포 Q (m, K+1)."""
    runs, m = ai_answers.shape
    q = np.full((m, n_options + 1), 0.05)
    for r in ai_answers:
        q[np.arange(m), r] += 1
    return q / q.sum(axis=1, keepdims=True)


def ai_lr(responses, bands, prop, q, f=F_AI):
    """AI 우도비: '문항의 f 비율을 AI 답으로 채웠다' 대 '정직' 의 정확 p값.

    AI가 맞힌 문항, 틀린 문항 정보를 모두 쓴다. 사람에게 어렵고 AI에게 쉬운 문항을 맞히거나
    AI가 고르는 오답을 고를수록 통계량이 커진다.
    """
    values = np.log(f * q[None] + (1 - f) * prop) - np.log(prop)
    tail, iv = sum_tail(values, prop)
    return tail[bands, _observed_idx(iv, bands, responses)]


def ai_wrong_match(responses, key, bands, prop, ai):
    """AI 오답 일치: AI(대표 답)가 틀린 문항에서 같은 오답을 고른 수의 정확 검정. 설명하기 쉬운 보조 지표."""
    n = responses.shape[0]
    items = np.nonzero((ai != key) & (ai > 0))[0]
    if len(items) == 0:
        return np.zeros(n, dtype=np.int64), np.zeros(n), np.ones(n)
    p = prop[:, items, ai[items]]
    tail = tail_probs(p)
    obs = (responses[:, items] == ai[items]).sum(axis=1)
    return obs, p.sum(axis=1)[bands], tail[bands, obs]


def preknowledge(responses, key, leaked, n_options, include=None):
    """사전 유출: 유출 의심 문항에서 비정상적으로 많이 맞혔는가 (정확 p값).

    점수대는 유출되지 않은 문항 점수로 나눈다. 총점으로 나누면 유출로 오른 점수가 신호를 흡수한다.
    """
    correct = responses == key
    secure = np.setdiff1d(np.arange(responses.shape[1]), leaked)
    bands = score_bands(correct[:, secure].sum(axis=1))
    prop = option_probs(responses, bands, n_options, include)
    p = prop[:, leaked, key[leaked]]
    obs = correct[:, leaked].sum(axis=1)
    return obs, tail_probs(p)[bands, obs]


def response_time(times, n_iter=20):
    """응답시간 이상: 로그정규 모형 log t_ij = beta_j - tau_i + e (van der Linden 2006).

    문항별 표준화 잔차 제곱합을 양측 카이제곱으로 검정한다. 상측은 들쭉날쭉한 시간
    (특정 문항만 비정상적으로 빠름), 하측은 문항 난이도와 무관하게 일정한 시간(답을 받아 적음).
    반환: z (n, m), p (n,)
    """
    lt = np.log(np.where(times > 0, times, np.nan))
    tau = np.zeros(lt.shape[0])
    for _ in range(n_iter):
        beta = np.nanmean(lt + tau[:, None], axis=0)
        tau = np.nanmean(beta[None, :] - lt, axis=1)
        tau -= tau.mean()
    resid = lt - (beta[None, :] - tau[:, None])
    sd = np.nanstd(resid, axis=0, ddof=1)
    z = resid / sd
    chi = np.nansum(z ** 2, axis=1)
    df = np.isfinite(z).sum(axis=1) - 1

    # 시간-문항 연동: 정직하면 log t를 문항 시간강도 beta에 회귀한 기울기가 1.
    # 답을 받아 적으면 문항과 무관하게 시간이 일정해서 기울기가 0 근처로 떨어진다.
    w = np.where(np.isfinite(lt), 1 / sd[None, :] ** 2, 0.0)
    lt0 = np.nan_to_num(lt)
    bw = (w * beta).sum(axis=1) / w.sum(axis=1)
    dev = beta[None, :] - bw[:, None]
    sxx = (w * dev ** 2).sum(axis=1)
    slope = (w * dev * lt0).sum(axis=1) / sxx

    # 경험적 귀무분포 (Efron): 실제 데이터에서는 사람마다 시간 변동 폭이 원래 달라서 이론적 카이제곱이
    # 크게 틀린다 (실제 자격시험 데이터에서 정상 응시자의 31%가 p<0.01). 다수가 정직하다는 가정 아래
    # 응시자 분포의 중앙값과 MAD로 표준화한 뒤 극단값만 본다.
    with np.errstate(divide="ignore", invalid="ignore"):
        stats = [np.log(chi / df), slope]
    p = np.ones(len(chi))
    for s in stats:
        ok = np.isfinite(s) & (df > 0)
        if ok.sum() < 30:
            continue
        med = np.median(s[ok])
        mad = 1.4826 * np.median(np.abs(s[ok] - med))
        if mad <= 0:
            continue
        zz = np.where(ok, (s - med) / mad, 0.0)
        p = np.minimum(p, np.array([math.erfc(abs(v) / math.sqrt(2)) for v in zz]))
    return z, np.minimum(1.0, 2 * p), tau  # 두 통계량 본페로니. tau: 클수록 빠름


def speed_score(tau, scores):
    """실력 대비 속도: 점수가 같은 사람들보다 얼마나 빠른가 (강건 z, 클수록 빠름).

    실제 자격시험 데이터에서 업체가 사전유출로 의심한 응시자를 가장 잘 가려냈다 (AUC 약 0.77).
    한계: 실제 속도 분포는 빠른 쪽 꼬리가 두꺼워 p값이 맞지 않는다 (정상 응시자 7~8%가 p<0.01).
    그래서 판정에는 쓰지 않고 조사 우선순위 점수로만 낸다.
    """
    slope, intercept = np.polyfit(scores, tau, 1)
    resid = tau - (slope * scores + intercept)
    med = np.median(resid)
    mad = 1.4826 * np.median(np.abs(resid - med))
    return (resid - med) / mad if mad > 0 else np.zeros_like(resid)


def score_gain(scores, prior):
    """이전 성적(모의고사 등) 대비 급상승. 회귀 잔차의 z값과 단측 p값."""
    slope, intercept = np.polyfit(prior, scores, 1)
    resid = scores - (slope * prior + intercept)
    sd = resid.std(ddof=2)
    z = resid / sd if sd > 0 else np.zeros_like(resid)
    p = np.array([0.5 * math.erfc(v / math.sqrt(2)) for v in z])
    return z, p


def guttman_gstar(correct):
    """정규화 거트만 오류 G* (0~1). 보조 지표 (p값 없음)."""
    order = np.argsort(-correct.mean(axis=0), kind="stable")
    c = correct[:, order].astype(np.int64)
    m = c.shape[1]
    r = c.sum(axis=1)
    g = (c * np.cumsum(1 - c, axis=1)).sum(axis=1)
    denom = r * (m - r)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, g / np.maximum(denom, 1), np.nan)


# ---------------------------------------------------------------- 전체 분석

def analyze(responses, key, ai_runs=None, groups=None, prior=None, times=None, leaked=None,
            school=None, cross_room=False, alpha=0.05, n_bands=20):
    """모든 지표를 계산한다.

    ai_runs: {모델이름: (실행횟수, m) 응답 배열}
    groups: 고사실 등 그룹 라벨 (n,)
    school: 출신 학교·학원 라벨 (n,). 공통 오답 성향을 보정한다
    cross_room: groups가 있어도 고사실을 넘는 쌍(휴대폰 공유 등)을 LSH로 함께 찾는다
    prior: 이전 성적 (n,)
    times: 문항별 응답시간(초) (n, m), 없는 값은 nan
    leaked: 유출 의심 문항 번호(0부터) 리스트

    판정은 방법(family)마다 본페로니 보정: alpha / 그 방법의 검정 수.
    """
    responses = np.asarray(responses, dtype=np.int64)
    key = np.asarray(key, dtype=np.int64)
    n, m = responses.shape if responses.ndim == 2 else (0, 0)
    if responses.ndim != 2 or key.shape != (m,):
        raise ValueError("응답 행렬의 문항 수와 정답 수가 다릅니다")
    if (responses < 0).any() or (key < 1).any():
        raise ValueError("응답은 0(무응답) 이상, 정답은 1 이상이어야 합니다")
    if n < 2:
        raise ValueError("수험생이 2명 이상 필요합니다")
    n_options = int(max(responses.max(), key.max()))
    ai_runs = {name: np.asarray(r, dtype=np.int64).reshape(-1, m) for name, r in (ai_runs or {}).items()}
    if groups is not None:
        _, groups = np.unique(np.asarray(groups), return_inverse=True)
    if school is not None:
        _, school = np.unique(np.asarray(school), return_inverse=True)
    leaked = np.asarray(sorted(set(leaked)), dtype=np.int64) if leaked else None

    correct = responses == key
    scores = correct.sum(axis=1)
    bands = score_bands(scores, n_bands)
    prop = option_probs(responses, bands, n_options)
    qs = {name: ai_distribution(r, n_options) for name, r in ai_runs.items()}
    modal = {name: q[:, 1:].argmax(axis=1) + 1 for name, q in qs.items()}

    n_groups = int(groups.max()) + 1 if groups is not None else 0
    all_pairs = n * (n - 1) // 2

    def pair_families(p_keep, prop_, use_lsh=True):
        """쌍 검정 가족들 [(이름, 후보, 검정 수)]과 고사실 통계를 돌려준다.

        - 고사실내: groups가 있으면 같은 고사실의 모든 쌍 (school이 있으면 학교 보정)
        - 전체: groups가 없거나 cross_room이면 고사실을 넘는 쌍. 많으면 LSH 후보만 정밀 검정
        """
        probs = person_probs(bands, prop_, school, responses)
        fams, room = [], None
        if groups is not None:  # 고사실별 쌍만 만든다: O(n × 고사실 인원)
            ia, ib = within_group_pairs(groups)
            cand, all_p = explicit_pairs(responses, key, probs, ia, ib, p_keep)
            nt = len(ia)
            room = (np.bincount(groups[ia], minlength=n_groups),
                    np.bincount(groups[ia][all_p < ROOM_P], minlength=n_groups))
            fams.append(("고사실내", cand, nt))
        if groups is None or cross_room:
            nt = all_pairs - (fams[0][2] if fams else 0)
            if groups is None and school is None and all_pairs <= ALL_PAIRS_MAX:
                cand = pair_screen(responses, key, bands, prop_, None, p_keep=p_keep)[0]
            elif groups is None and all_pairs <= 2e6:
                ia, ib = np.triu_indices(n, 1)
                cand = explicit_pairs(responses, key, probs, ia, ib, p_keep)[0]
            elif use_lsh:
                ia, ib = lsh_candidates(responses, key, n_options)
                if groups is not None:
                    keep = groups[ia] != groups[ib]
                    ia, ib = ia[keep], ib[keep]
                # 판정 기준이 극도로 엄격(쌍 수십억)하므로 우도비 정밀 검정은 p<1e-4 후보만
                cand = explicit_pairs(responses, key, probs, ia, ib, min(p_keep, 1e-4))[0]
                out_lsh["candidates"] = len(ia)
            else:
                return fams, room
            fams.append(("전체", cand, nt))
        return fams, room

    out_lsh = {}
    # 2단계 추정: 부정행위자가 많으면 그들의 선택이 '흔한 선택'처럼 보여 스스로를 가린다.
    suspect = np.zeros(n, dtype=bool)
    for _, cand, _ in pair_families(PRELIM, prop, use_lsh=False)[0]:
        for a, b, *_ in cand:
            suspect[a] = suspect[b] = True
    for name in qs:
        suspect |= ai_lr(responses, bands, prop, qs[name]) < PRELIM
        suspect |= ai_wrong_match(responses, key, bands, prop, modal[name])[2] < PRELIM
    if leaked is not None:
        suspect |= preknowledge(responses, key, leaked, n_options)[1] < PRELIM
    include = None
    if suspect.any() and (~suspect).sum() >= 50:
        include = ~suspect
        prop = option_probs(responses, bands, n_options, include=include)

    out = {"scores": scores, "bands": bands, "gstar": guttman_gstar(correct), "warnings": [], "lsh": out_lsh,
           "n_excluded": int(suspect.sum()), "thresholds": {}, "flags": {}, "p": {}}

    # 답안 공유: 같은 오답 정확 검정으로 훑고, 후보 쌍은 우도비로 정밀 검정
    probs_of = person_probs(bands, prop, school, responses)
    families, room = pair_families(p_keep=1e-2, prop_=prop)
    pairs = []
    for label, cand, n_tests in families:
        thr = alpha / len(families) / (2 * max(n_tests, 1))  # 가족 수, 두 통계량(같은오답, 우도비)으로 나눔
        out["thresholds"][f"pair_{label}"] = thr
        for a, b, x, e, p_count in cand:
            pa, pb = probs_of(np.array([a, b]))
            p_lr = min(1.0, 2 * min(copy_lr_p(responses[a], responses[b], pb),
                                    copy_lr_p(responses[b], responses[a], pa)))
            pairs.append((a, b, x, e, p_count, p_lr, p_count < thr or p_lr < thr, label))
    out["pairs"] = pairs
    out["n_pair_tests"] = sum(f[2] for f in families)
    flagged_pairs = [(p[0], p[1]) for p in pairs if p[6]]
    out["cluster"] = clusters(n, flagged_pairs)
    pair_flag = np.zeros(n, dtype=bool)
    for a, b in flagged_pairs:
        pair_flag[a] = pair_flag[b] = True
    out["flags"]["답안공유"] = pair_flag

    if room is not None:
        room_pairs, room_hits = room
        room_p = np.array([binom_sf(int(h), int(t), ROOM_P) for h, t in zip(room_hits, room_pairs)])
        out["rooms"] = (room_pairs, room_hits, room_p)
        out["thresholds"]["room"] = alpha / len(room_p)
        out["room_flag"] = room_p < out["thresholds"]["room"]
        if out["room_flag"].mean() > ROOM_WARN:
            out["warnings"].append(
                f"고사실 {out['room_flag'].mean():.0%}가 판정됐습니다. 고사실 전체 부정보다 같은 학교·학원 "
                "학생이 함께 앉아 생기는 공통 오답일 가능성이 큽니다. school 열을 지정해 다시 분석하세요.")

    p_fit = person_fit(responses, bands, prop)
    out["p"]["이상응답"] = p_fit
    out["thresholds"]["이상응답"] = alpha / n
    out["flags"]["이상응답"] = p_fit < alpha / n

    if qs:
        # 점수대는 총점 기준이어야 한다. 이전 성적처럼 잡음 큰 기준으로 나누면 점수대 안 실력 차이 때문에
        # 문항 간 독립 가정이 깨져 오탐이 난다 (가상 데이터에서 1만 명당 6명 오탐 확인 후 제거).
        lr_p = np.column_stack([ai_lr(responses, bands, prop, qs[name]) for name in qs])
        thr = alpha / (2 * n * len(qs))
        wm = [ai_wrong_match(responses, key, bands, prop, modal[name]) for name in qs]
        out["ai"] = {"models": list(qs), "lr_p": lr_p,
                     "wrong_obs": np.column_stack([w[0] for w in wm]),
                     "wrong_exp": np.column_stack([w[1] for w in wm]),
                     "wrong_p": np.column_stack([w[2] for w in wm])}
        out["thresholds"]["ai"] = thr
        out["flags"]["AI"] = (lr_p < thr).any(axis=1) | (out["ai"]["wrong_p"] < thr).any(axis=1)

    if leaked is not None:
        obs, p = preknowledge(responses, key, leaked, n_options, include)
        out["preknowledge"] = (obs, p)
        out["thresholds"]["사전유출"] = alpha / n
        out["flags"]["사전유출"] = p < alpha / n

    if times is not None:
        times = np.asarray(times, dtype=float)
        if times.shape != responses.shape:
            raise ValueError("응답시간 행렬 크기가 응답 행렬과 다릅니다")
        _, p, tau = response_time(times)
        out["speed_z"] = speed_score(tau, scores)  # 판정 아님, 우선순위 점수
        out["p"]["응답시간"] = p
        out["thresholds"]["응답시간"] = alpha / n
        out["flags"]["응답시간"] = p < alpha / n

    if prior is not None:
        z, p = score_gain(scores, np.asarray(prior, dtype=float))
        out["gain"] = (z, p)
        out["thresholds"]["급상승"] = alpha / n
        out["flags"]["급상승"] = p < alpha / n

    out["any_flag"] = np.logical_or.reduce(list(out["flags"].values()))
    return out
