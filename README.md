# exam-forensics

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.4.0-blue.svg)](pyproject.toml)

시험 응답 데이터를 분석해 부정행위가 의심되는 응시자를 찾아내는 통계 도구입니다. 답안 공유, AI를 이용한 풀이, 문제 사전 유출, 비정상적인 풀이 시간을 탐지하며, 모든 분석은 수험생 데이터를 외부로 보내지 않고 로컬에서 처리합니다.

---

## 목차

- [특징](#특징)
- [설치](#설치)
- [사용법](#사용법)
- [입력 형식](#입력-형식)
- [출력](#출력)
- [AI 답안 수집](#ai-답안-수집)
- [동작 방식](#동작-방식)
- [검증](#검증)
- [프로젝트 구조](#프로젝트-구조)
- [알려진 한계](#알려진-한계)
- [개발](#개발)
- [기여](#기여)
- [라이선스](#라이선스)

---

## 특징

### 답안 공유 탐지
- 같은 오답이 겹치는 정도로 모든 응시자 쌍을 1차 검사
- 의심 쌍은 우도비 검정으로 재확인 (어려운 문항의 정답, 드문 오답이 겹칠수록 가중)
- 판정된 쌍을 연결해 **공모 집단** 단위로 묶어서 표시
- 고사실별로 유사 쌍이 기대보다 많은지 검사해 고사실 단위 부정도 탐지

### 전국 규모 분석
- `--cross-room` 옵션으로 고사실을 넘나든 답안 공유(휴대폰 등)까지 탐지
- **MinHash LSH**로 후보 쌍을 먼저 추려 10만 명(50억 쌍)도 약 90초에 처리
- 고사실 안 쌍은 고사실 단위로만 계산해 응시자 수에 비례하는 비용으로 동작

### AI 부정 탐지
- 여러 AI 모델이 같은 시험을 푼 답안과 비교
- 응답이 사람들의 일반적인 답보다 AI 쪽으로 기울었는지 우도비로 검정
- AI 특유의 오답을 그대로 따라 고른 개수를 별도로 집계 (설명하기 쉬운 근거)
- OpenAI 호환 API, Anthropic API, 로컬 모델(Ollama)로 AI 답안 수집 지원

### 사전 유출 · 응답시간 · 이상응답
- **사전 유출**: 유출 의심 문항의 정답률을 유출되지 않은 문항 점수 기준으로 비교
- **응답시간**: 문항 난이도와 무관하게 일정한 시간(답을 받아 적는 경우)이나 지나치게 불규칙한 시간 탐지
- **실력 대비 풀이 속도**: 비슷한 점수대보다 유난히 빠른 응시자를 순위 점수로 제공
- **이상응답**: 같은 점수대에서 보기 드문 응답 패턴 탐지
- **성적 급상승**: 이전 성적 대비 비정상적으로 높은 점수 탐지

### 오탐 통제
- 모든 확률을 근사 없이 정확하게 계산
- 검사 수만큼 기준을 높이는 본페로니 보정으로 정상 응시자 오지목 확률을 항목별 5% 이하로 유지
- 부정행위자가 많을 때 기준이 오염되지 않도록 1차 의심자를 빼고 재추정
- 같은 학교 학생이 모여 앉아 생기는 공통 오답은 `school` 열로 보정하고, 의심 상황에서는 자동 경고

---

## 설치

Python 3.9 이상과 numpy가 필요합니다.

```bash
git clone https://github.com/main-7o/exam-forensics.git
cd exam-forensics
pip install numpy
```

Anthropic API로 AI 답안을 수집하려면 추가로 설치합니다.

```bash
pip install anthropic
```

---

## 사용법

### 데모

```bash
python -m exam_forensics demo
```

부정행위자를 섞은 가상 시험을 만들어 분석하고, 유형별로 몇 명을 찾아냈는지와 정상 응시자 오지목 수를 보여 줍니다.

### 분석

```bash
python -m exam_forensics analyze responses.csv --key key.csv --out report
```

| 옵션 | 설명 |
|------|------|
| `--key` | 정답 파일 (필수) |
| `--ai` | AI 답안 파일. 지정하면 AI 부정 분석 |
| `--times` | 문항별 풀이 시간 파일. 지정하면 응답시간 분석 |
| `--leaked` | 유출 의심 문항 목록 (예: `q3,q7`). 지정하면 사전 유출 분석 |
| `--cross-room` | 고사실을 넘나든 답안 공유까지 분석 |
| `--alpha` | 유의수준 (기본 0.05) |
| `--out` | 결과 폴더 (기본 `report`) |

### 예제 데이터 생성

```bash
python -m exam_forensics simulate --out sample
python -m exam_forensics analyze sample/responses.csv --key sample/key.csv \
    --ai sample/ai.csv --times sample/times.csv --out report
```

---

## 입력 형식

### responses.csv

한 행에 응시자 한 명이며, 문항 열에는 선택한 보기 번호를 적습니다. 무응답은 빈칸 또는 `0`입니다.

```
id,group,school,prior,q1,q2,q3,...
s0,room01,schoolA,72.5,3,1,5,...
```

| 열 | 필수 | 설명 |
|----|:---:|------|
| `id` | O | 응시자 식별자 |
| `group` | | 고사실. 같은 고사실 안에서 쌍을 비교하고 고사실 단위로도 분석 |
| `school` | | 출신 학교·학원. 같은 곳에서 배운 학생들의 공통 오답 보정 |
| `prior` | | 모의고사 등 이전 성적 |
| 그 외 | O | 문항별 응답 |

### 기타 파일

| 파일 | 형식 |
|------|------|
| `key.csv` | 문항 열과 같은 헤더에 정답 한 행 |
| `ai.csv` | `model`, `run`, `collected_at`과 문항 열. 같은 모델의 여러 회차는 하나로 합산 |
| `times.csv` | `id`와 문항 열. 값은 풀이 시간(초), 컴퓨터 기반 시험에서 수집 |

---

## 출력

`--out`으로 지정한 폴더에 생성됩니다.

| 파일 | 내용 |
|------|------|
| `examinees.csv` | 응시자별 점수, 항목별 p값, 공모 집단 번호, 판정 근거, 조사 대상 여부 |
| `pairs.csv` | 응답이 비정상적으로 비슷한 응시자 쌍과 검정 결과 |
| `rooms.csv` | 고사실별 유사 쌍 수와 판정 결과 (`group` 열이 있을 때) |

분석 결과는 부정행위의 증거가 아니라 조사가 필요한 대상을 추리는 참고 자료입니다. 최종 판단은 감독 기록, 진술 등 다른 근거와 함께 사람이 내려야 합니다.

---

## AI 답안 수집

`collect-ai`는 시험 문제를 AI 모델에 보내 답을 모아 `ai.csv`에 기록합니다. 수험생 데이터는 사용하지 않습니다.

```bash
# Anthropic API (ANTHROPIC_API_KEY 필요)
python -m exam_forensics collect-ai questions.json --provider anthropic --model claude-opus-5-5 --runs 5

# OpenAI 호환 API (OpenAI, Gemini, DeepSeek 등)
python -m exam_forensics collect-ai questions.json --provider openai \
    --base-url https://api.openai.com/v1 --api-key-env OPENAI_API_KEY --model <모델>

# 로컬 모델 (Ollama, 인터넷 불필요)
python -m exam_forensics collect-ai questions.json --provider openai \
    --base-url http://localhost:11434/v1 --model <모델>
```

`questions.json`의 `id`는 응답 파일의 문항 열 이름과 같아야 합니다.

```json
[{"id": "q1", "question": "다음 중 ...", "choices": ["보기1", "보기2", "보기3", "보기4", "보기5"]}]
```

- 모델별로 실행하면 같은 파일에 누적되고, 수집 시각이 함께 기록됨
- AI 모델은 수시로 업데이트되므로 시험 직후 수집 권장
- 그림·도표가 포함된 문항은 아직 미지원

---

## 동작 방식

```
┌─────────────────┐     ┌──────────────────────┐     ┌──────────────────┐
│ responses.csv   │     │ 점수대 기준 모델      │     │ 검정              │
│ key.csv         │────▶│ (점수대별 실제        │────▶│ 답안 공유 · AI    │
│ times.csv       │     │  선택 비율)           │     │ 사전 유출 · 시간  │
│ ai.csv          │     │                      │     │ 이상응답 · 급상승  │
└─────────────────┘     └──────────┬───────────┘     └────────┬─────────┘
                                   │                          │
                        1차 의심자 제외 후 재추정              │ 본페로니 보정
                                   │                          ▼
                                   └──────────────▶  ┌──────────────────┐
                                                     │ report/          │
┌─────────────────┐                                  │ examinees.csv    │
│ questions.json  │── collect-ai ──▶ ai.csv          │ pairs.csv        │
│ (문제만 전송)    │   (분석과 분리)                   │ rooms.csv        │
└─────────────────┘                                  └──────────────────┘
```

1. 응시자를 총점 기준 점수대로 나누고, 점수대별로 각 문항의 실제 선택 비율을 계산
2. 이 비율을 "정상 응시자라면 이렇게 답한다"는 기준으로 삼아 항목별 확률을 정확히 계산
3. 1차로 강하게 의심되는 응시자를 빼고 기준을 다시 추정 (부정행위자가 기준을 오염시키는 것 방지)
4. 검사 수만큼 기준을 높여 최종 판정
5. 네트워크를 쓰는 코드는 `collect-ai`에만 있으며, 분석 단계와 완전히 분리

---

## 검증

공개된 실제 자격시험 데이터(Cizek & Wollack, 2017, 약 3,200명)로 확인했습니다. 시험 주관사가 사전 유출 의심자로 분류한 94명을, 그 분류를 모르는 상태에서 얼마나 찾아내는지 측정한 결과입니다.

| 항목 | AUC | 오탐률 5%에서의 탐지율 |
|------|-----|-----|
| 답안 공유 | 0.63 | 29% |
| 실력 대비 풀이 속도 | 0.77 | 47% |
| 항목 종합 | **0.92** | **78%** |

가상 데이터에서는 부정행위자가 없는 시험 10만 명분에서 오지목 1명, 문항을 모두 베낀 응시자와 AI를 모든 문항에 쓴 응시자는 각각 90%를 찾아냈습니다.

자세한 실험 조건과 한계는 [docs/validation.md](docs/validation.md)에 있습니다.

---

## 프로젝트 구조

```
exam-forensics/
├── exam_forensics/
│   ├── __main__.py      # 명령줄 도구 (analyze, collect-ai, simulate, demo)
│   ├── core.py          # 모든 통계 검정과 분석 흐름
│   ├── ai_collect.py    # AI 답안 수집 (네트워크를 쓰는 유일한 모듈)
│   └── simulate.py      # 검증용 가상 시험 데이터 생성
├── benchmarks/
│   ├── cizek_wollack.py # 실제 시험 데이터 검증
│   └── plant_real.py    # 실제 응답에 가상 부정행위를 삽입하는 실험
├── tests/
│   └── test_core.py
├── docs/
│   └── validation.md    # 검증 결과 상세
├── pyproject.toml
├── LICENSE
└── README.md
```

---

## 한계

- **실데이터 검증 범위**: 아직 데이터셋 하나로만 검증. 실제 적용 전 해당 기관의 과거 데이터로 오탐률 확인 권장
- **AI 일부 사용**: AI를 일부 문항에만 쓴 경우 탐지율이 낮음 (문항의 50% 사용 시 약 10%)
- **AI 모델 차이**: 응시자가 쓴 AI와 수집에 쓴 AI가 같다는 보장 없음
- **학교 효과와 집단 부정**: 고사실 전체의 공모와 같은 학교 출신의 공통 오답은 통계만으로 구분 불가
- **미지원 기능**: 정답 수정 흔적(OMR 지운 자국) 분석, 컴퓨터 적응형 시험(CAT), 그림 문항의 AI 수집

---

## 개발

### 테스트

```bash
python -m unittest discover -s tests -t .
```

### 실데이터 검증 재현

```bash
python benchmarks/cizek_wollack.py <데이터 폴더>
python benchmarks/plant_real.py <데이터 폴더>
```

데이터는 [공개본](https://cengiz.me/posts/xgboost/)에서 받을 수 있으며, 사용 시 원 출처(Cizek & Wollack, 2017)를 밝혀야 합니다.

### 주요 설정값 (core.py)

| 상수 | 기본값 | 설명 |
|------|------|------|
| `PRELIM` | `1e-3` | 기준 재추정 시 제외할 1차 의심자 기준 |
| `F_COPY` | `0.5` | 답안 공유 검정에서 가정하는 베낀 문항 비율 |
| `F_AI` | `0.7` | AI 검정에서 가정하는 AI 사용 문항 비율 |
| `SCHOOL_KAPPA` | `2.0` | 학교 보정 강도 |
| `ALL_PAIRS_MAX` | `5e7` | 모든 쌍을 직접 비교하는 최대 쌍 수 (초과 시 LSH 사용) |
| `ROOM_WARN` | `0.05` | 판정된 고사실 비율이 이를 넘으면 경고 |

---

## 기여

이슈와 PR 모두 환영합니다.

### 이슈 등록 시 포함해 주세요

- Python, numpy 버전
- 실행한 명령어와 오류 메시지
- 가능하다면 문제를 재현할 수 있는 최소한의 데이터 (실제 수험생 정보는 제외)

### 개발 가이드라인

- 새 검정을 추가할 때는 부정행위자가 없는 가상 데이터에서 오탐이 늘지 않는지 먼저 확인
- 실제 데이터로 검증한 결과가 있다면 `docs/validation.md`에 함께 기록
- 분석 코드에는 네트워크를 쓰는 코드를 추가하지 않음

---

## 라이선스

[MIT](LICENSE) © 2026 MAIN

이 도구의 분석 결과만으로 응시자를 부정행위자로 단정해서는 안 됩니다. 결과의 활용과 그에 따른 판단의 책임은 사용자에게 있습니다.
