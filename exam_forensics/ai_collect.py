"""AI 응답 수집기. 시험 '문제'만 AI에 보내고, 수험생 데이터는 절대 쓰지 않는다.

분석(analyze)은 이 모듈을 import하지 않는다. 네트워크는 이 모듈에서만 쓴다.

문제 파일(JSON):
[{"id": "q1", "question": "다음 중 ...", "choices": ["보기1", "보기2", "보기3", "보기4", "보기5"]}, ...]
"""
import csv
import datetime
import json
import os
import re
import urllib.request

PROMPT = ("다음 객관식 문제를 풀고 정답 번호를 고르세요.\n"
          "풀이는 짧게 써도 되지만, 마지막 줄은 반드시 '정답: 번호' 형식으로 쓰세요.\n\n{body}")
ANSWER = re.compile(r"정답\s*[:：]\s*\(?([1-9])")


def format_question(q):
    lines = [q["question"], ""] + [f"{i + 1}. {c}" for i, c in enumerate(q["choices"])]
    return PROMPT.format(body="\n".join(lines))


def parse_answer(text, n_choices):
    """마지막 '정답: N'을 읽는다. 없거나 범위 밖이면 0(무응답)."""
    found = ANSWER.findall(text or "")
    if not found:
        return 0
    k = int(found[-1])
    return k if 1 <= k <= n_choices else 0


def ask_openai_compatible(prompt, model, base_url, api_key):
    """OpenAI 호환 /chat/completions (OpenAI, Gemini, DeepSeek, Ollama 등)."""
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}]}).encode(),
        headers={"Content-Type": "application/json",
                 **({"Authorization": f"Bearer {api_key}"} if api_key else {})})
    with urllib.request.urlopen(req, timeout=600) as r:
        body = json.load(r)
    return body["choices"][0]["message"]["content"]


def ask_anthropic(prompt, model, client):
    """Anthropic 공식 SDK. 거절 시 서버 측 대체 모델로 넘어가도록 fallbacks를 켠다."""
    response = client.beta.messages.create(
        model=model,
        max_tokens=16000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason == "refusal":
        return ""
    return "".join(b.text for b in response.content if b.type == "text")


def collect(questions, provider, model, runs=5, base_url=None, api_key_env=None, log=print):
    """runs번 반복해 문항별 답을 모은다. 반환: (실행마다 {문항id: 답} 리스트, 수집 시각)."""
    if provider == "anthropic":
        import anthropic  # 선택 의존성: pip install anthropic
        client = anthropic.Anthropic()
        ask = lambda p: ask_anthropic(p, model, client)  # noqa: E731
    elif provider == "openai":
        if not base_url:
            raise ValueError("openai 호환 공급자는 --base-url 이 필요합니다")
        key = os.environ.get(api_key_env) if api_key_env else None
        ask = lambda p: ask_openai_compatible(p, model, base_url, key)  # noqa: E731
    else:
        raise ValueError(f"알 수 없는 공급자: {provider}")

    collected_at = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
    results = []
    for run in range(runs):
        answers = {}
        for q in questions:
            try:
                answers[q["id"]] = parse_answer(ask(format_question(q)), len(q["choices"]))
            except Exception as e:  # 한 문항 실패로 전체 수집을 버리지 않는다
                log(f"  {model} {run + 1}회차 {q['id']} 실패: {e}")
                answers[q["id"]] = 0
        results.append(answers)
        log(f"{model} {run + 1}/{runs}회 완료")
    return results, collected_at


def append_csv(path, model, results, collected_at, item_ids):
    """ai.csv에 행을 추가한다 (model, run, collected_at, 문항...). 파일이 있으면 이어 쓴다."""
    exists = os.path.exists(path)
    if exists:
        with open(path, newline="", encoding="utf-8-sig") as f:
            header = next(csv.reader(f))
        if header[3:] != item_ids:
            raise ValueError(f"{path}의 문항 열이 문제 파일과 다릅니다")
    # 이어 쓸 때 utf-8-sig를 쓰면 파일 중간에 BOM이 들어간다
    with open(path, "a", newline="", encoding="utf-8" if exists else "utf-8-sig") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["model", "run", "collected_at"] + item_ids)
        for i, ans in enumerate(results):
            w.writerow([model, i + 1, collected_at] + [ans[q] for q in item_ids])


def load_questions(path):
    with open(path, encoding="utf-8-sig") as f:
        qs = json.load(f)
    for q in qs:
        if not {"id", "question", "choices"} <= q.keys() or len(q["choices"]) < 2:
            raise ValueError(f"문제 형식 오류: {q.get('id')}")
    return qs
