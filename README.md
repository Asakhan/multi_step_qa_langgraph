# 다단계 수치 추론 문서 QA — LangGraph 오케스트레이션 구조 실험 (재현 패키지)

석사논문 3장 및 부록 A 실험의 **LangGraph 측 구현·재현 패키지**다. 하나의 상태 그래프
(`AgentState`)·동일 도구·동일 저자확정 프롬프트 위에서 여러 오케스트레이션 구조를 구현하고,
동일한 60문항(FinQA 20 + K-DART-QA 40, 모두 self-contained)에 대해 각 문항을 3회 반복
실행한 뒤 보정 채점기로 채점하고 통계 분석까지 수행한다.

이 저장소는 다음 두 가지를 그대로 재현하는 데 초점을 맞춘다.

- **예비 실험 (부록 A)**: LangGraph 단일 프레임워크 위 다섯 구조 `S0·S1·S2·S3·S4` 비교.
- **본 실험 (3장)**: `실험 2`(LangGraph 검증 항상)와 `실험 3`(LangGraph 동적 신뢰도 캐스케이드).

> 본 실험의 `실험 1`(CrewAI 검증 구조)은 **별도 저장소**에서 구현된다. 이 저장소는
> 두 프레임워크 비교의 LangGraph 쪽(실험 2·3)과 예비 실험만 담당한다. 두 프레임워크는
> 동일한 self-contained 컨텍스트·동일 채점기·동일 모델(gemini-2.5-flash)로 통제된다.

---

## 1. 논문 ↔ 코드 매핑

구조 식별자는 코드 전반에서 `s0`~`s5` 로 쓰인다. 논문 표기와의 대응은 다음과 같다.

| 코드 | 논문 위치 | 워크플로 | 모델(노드) | 비고 |
|---|---|---|---|---|
| `s0` | 예비 A — 저비용 기준선 | `retrieve→generate→END` | flash | baseline |
| `s1` | 예비 A — 상한(상위 모델) | s0와 동일 그래프, generate 모델만 교체 | gemini-pro | quota 미완(분석 제외) |
| `s2` | 예비 A + **본실험 실험 2** | `retrieve→generate→verify`(FAIL시 1회 재생성) | flash + verify=flash | 검증 항상 |
| `s3` | 예비 A — 이종 다수결 | `retrieve→[3 gen 병렬]→aggregate` | flash + gpt-4o-mini + claude-haiku | voting@3 |
| `s4` | 예비 A — 질의 분해 | `decompose→solve→aggregate→END` (sub-q ≤5) | flash(3역할) | 단계 분해 |
| `s5` | **본실험 실험 3** | `generate(confidence)→gate→{stop:END / escalate:verify}` | flash + verify=flash | 동적 신뢰도 캐스케이드 |

- **구조 간 차이는 노드·엣지 구성으로만** 발생한다(프롬프트·도구·상태 스키마 동일).
- 본 실험(실험 2·3)은 논문 3.2.1 통제에 따라 **gemini-2.5-flash 단일 모델**만 사용한다 →
  `GOOGLE_API_KEY` 하나로 실행 가능. 예비 실험의 `s1/s3`만 OpenAI·Anthropic 키가 추가로 필요하다.
- `temperature=0`(결정적 디코딩)으로 모든 호출을 통제한다.

> 가설 표기 주의: 논문 본문의 H1·H2·H3 는 **CrewAI ↔ LangGraph 프레임워크 비교**(교차 저장소)다.
> 이 저장소의 `src/analysis.py` 가 계산하는 H1·H2 라벨은 **LangGraph 내부 구조 비교**(예비
> 실험 진단용: S2·S3 의 TPS vs S0, S4 vs S0)이며, 프레임워크 비교(논문 H1·H2)는 두 저장소의
> 구조별 지표(TSR·Token·TPS)를 합쳐 수행한다.

---

## 2. 저장소 구조

```
config/
  models.yaml       구조→노드→논리모델 매핑, 논리모델→(env, 단가키, provider)
  experiment.yaml   통제 변수(반복 3회, top_k, 채점 허용오차, 게이트 임계값 등)
  pricing.yaml      모델별 단가표(USD/1M tokens)
data/
  finqa/finqa.jsonl            FinQA 영어 20문항 (self-contained, 커밋된 정본)
  kdart/kdart_finqa.jsonl      K-DART-QA 한국어 40문항 (self-contained, 커밋된 정본)
  kdart/kdart_qa_finqa_format.json   K-DART 원천(참고)
src/
  config.py         .env/yaml 로딩·검증, 경로 상수, RESULTS_DIR 오버라이드
  datasets.py       60문항을 공통 Question 레코드로 로딩
  tools.py          retrieve(self-contained 컨텍스트) + 안전 산술 평가 calc
  llm_clients.py    provider별 chat 모델 + 토큰/단가 계측 래퍼(temperature=0)
  state.py          단일 AgentState(병렬 필드는 reducer 누적)
  prompt_loader.py  저자확정 프롬프트 로더(AUTHOR-LOCKED 마커 제거)
  metrics.py        보정 채점기 is_correct + 비용/TSR/TPS 산출
  structures/       s0~s5.py + common.py(노드 빌더) + aggregation.py(S3 다수결)
  prompts/          저자 확정 시스템 프롬프트(*.txt, 변경 금지)
  runner.py         단일 구조 실행 엔진(전수 로깅, idempotent resume, 토큰 검산)
  orchestrator.py   구조들을 독립 subprocess로 순차 실행 + 최종 분석/메일
  build_index.py    데이터 준비/검증 진입점(키 불필요)
  rejudge.py        저장된 raw_runs 를 모델 재호출 없이 보정 채점기로 재채점
  analysis.py       McNemar / 부트스트랩 CI / 효과크기 / 파레토
  reporter.py, notify.py   진행 워처·메일(선택)
results/<sX>/       raw_runs.jsonl, metrics.json, failures.csv   (gitignore — 재현 산출물)
results/analysis/   analysis.json, pareto.png
run.sh              진입점
tests/              pytest (채점기 단위 테스트)
```

> `results*/`, `logs/`, `*.tar.gz`, `*.bak`, `data/chroma/`, `data/finqa/FinQA-main/` 는
> `.gitignore` 대상이다(재현 가능 산출물·대용량 원천). 데이터셋 정본 jsonl 은 커밋되어 있다.

---

## 3. 데이터셋

| 데이터셋 | 언어 | 문항 | 유형 | 포맷 |
|---|---|---|---|---|
| FinQA[8] | en | 20 | T3(단일근거) 10 + T4(다단계) 10 | self-contained `context_chunks`(text_N/table_N) |
| K-DART-QA[14] | ko | 40 | T1 5 · T2 10 · T3 10 · T4 10 · T5 5 | self-contained `context_chunks`(text_N/table_N) |

- 두 데이터셋 모두 **문항이 근거 문서를 통째로 포함**한다. 따라서 전역 검색 인덱스·임베딩이
  필요 없다. K-DART 는 전체 청크를 그대로 투입하고, FinQA 는 문항별 컨텍스트에 대해
  코사인 top_k=50 검색을 하되 컨텍스트 수(25~38)가 top_k 이하라 실질적으로 전체를 반환한다
  (임베딩 호출 미발생). → 두 데이터셋이 동일 파이프라인으로 대칭 비교된다.
- **K-DART-QA 구축 파이프라인**(DART 원문 수집→청크→문항 작성·검수→난이도 calibration)은
  별도 저장소에 공개되어 있다: <https://github.com/Asakhan/kdart_qa> (논문 §3.3.4, Apache-2.0).
  이 저장소의 `data/kdart/kdart_finqa.jsonl` 은 그 산출물(실험에 쓰인 정본)이다.

---

## 4. 설치

```bash
git clone git@github.com:Asakhan/multi_step_qa_langgraph.git
cd multi_step_qa_langgraph
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.11/3.12, LangChain 0.3.x / LangGraph 0.2.x
cp .env.example .env                      # 그 후 .env 값을 채운다 (5절)
```

데이터는 이미 커밋되어 있으므로 준비가 필요 없다. 다만 60문항이 정상 로드되는지 검증할 수 있다(키 불필요):

```bash
./run.sh build-index        # finqa 재생성(원천 있을 때) + 60문항(finqa 20 + kdart 40) 로드 검증
```

---

## 5. .env 작성

코드는 시작 시 `.env` 를 로드하고, 실행할 구조가 요구하는 키가 비면 **어떤 키가 빠졌는지
알려주며 즉시 중단**한다(빠른 실패). 실험별 최소 키는 다음과 같다.

| 실행 대상 | 필수 키 |
|---|---|
| **본실험 실험 2·3** (`s2`, `s5`) | `GOOGLE_API_KEY`, `MODEL_GEMINI_FLASH`(기본 gemini-2.5-flash) |
| 예비 실험 `s1` | + `MODEL_GEMINI_PRO` |
| 예비 실험 `s3` | + `OPENAI_API_KEY`, `MODEL_GPT4O_MINI`, `ANTHROPIC_API_KEY`, `MODEL_CLAUDE_HAIKU` |
| 메일 진행보고(선택) | `SMTP_HOST/PORT/USER/PASSWORD`, `MAIL_FROM`, `MAIL_TO`, `SEND_EMAIL=1` |

선택 키: `S5_THRESHOLD`(실험 3 게이트 임계값 오버라이드), `COST_CAP_USD`(누적 비용 상한
초과 시 일시정지+경고), `SEND_EMAIL=0`(메일 비활성화). 모델 문자열·단가·매핑은 코드에
하드코딩하지 않고 `config/*.yaml`·`.env` 에서만 읽는다.

> 본 실험은 self-contained 컨텍스트라 임베딩·ChromaDB 가 필요 없다 → `OPENAI_API_KEY`·
> `EMBEDDING_MODEL`·`DART_API_KEY` 는 실험 2·3 에 불필요하다.

---

## 6. 재현 절차

각 실행은 `RESULTS_DIR` 로 결과 디렉토리를 분리하는 것을 권장한다(예비/본/스윕 구분).
모든 실행은 `(구조, 문항, 반복)` 단위로 `raw_runs.jsonl` 에 append 되며, 중단 후 재실행하면
완료분은 건너뛰고 이어서 진행한다(idempotent). 실패(run_error) run 은 자동 재시도한다.

### 6.0 빠른 동작 검증 (GOOGLE 키만, 수 분)

```bash
./run.sh smoke                       # 본실험(s2 s5) × 데이터셋당 2문항 × 3반복, 전경 실행
./run.sh smoke s0 s2 s3 s4           # 예비실험까지 점검(OpenAI/Anthropic 키 필요)
```

### 6.1 본 실험 — 실험 2 & 실험 3 (gemini-2.5-flash 단일)

```bash
# 실험 2(검증 항상=s2) + 실험 3(동적 캐스케이드=s5), 각 60문항 × 3반복 = 360회
RESULTS_DIR=results_main ./run.sh s2 s5            # 백그라운드(setsid) 실행
RESULTS_DIR=results_main ./run.sh status           # 진행 확인(pid + 로그 tail + 구조별 완료 수)
RESULTS_DIR=results_main ./run.sh analysis         # 통계 분석 + 파레토
```

**실험 3 게이트 임계값 스윕**(논문 3.4.3 — 검증 발동률 대 TPS·TSR 운영 곡선):

```bash
for T in 0.5 0.6 0.7 0.8 0.9; do
  S5_THRESHOLD=$T RESULTS_DIR=results_exp3_t$T ./run.sh s5
done
```

게이트 신호는 모델 자기보고 confidence(`generate_s5.txt`)를 1차로 쓰고, 보조 신호로
**추론 답 vs 코드 실행 답 불일치**(`expressions` 재실행)가 감지되면 유효 confidence 를 0 으로
강제 하향해 검증으로 에스컬레이션한다. `confidence`(보정 후)·`confidence_self_report`(원자기보고)·
`code_conflict`·`gate_decision`·`escalated` 는 H3 보조가설(게이트 유효성) 분석용으로 전수 기록된다.
self-consistency 는 temperature=0 에서 작동하지 않으므로 게이트 신호로 쓰지 않는다(논문 3.1).

### 6.2 예비 실험 — S0~S4 (부록 A)

```bash
# S0~S4 전부(OpenAI·Anthropic 키 필요), 각 60문항 × 3반복
RESULTS_DIR=results_pre ./run.sh all --run-s0     # all = s1 s2 s3 s4, --run-s0 으로 s0 포함
RESULTS_DIR=results_pre ./run.sh status
RESULTS_DIR=results_pre ./run.sh analysis
```

> 보정 채점기는 `src/metrics.py` 에 이미 반영되어 있으므로 **새로 실행한 run 은 이미 보정
> 채점된다**. 부록 A.2 의 "1차 채점 → 보정 재채점" 과정 자체를 재현하려면(저장된 구
> raw 를 재채점) 6.4 의 rejudge 를 사용한다.

### 6.3 분석 산출물

`./run.sh analysis`(= `python -m src.analysis`)는 `RESULTS_DIR/analysis/` 에 다음을 쓴다.

- `analysis.json` — 구조별 포인트(TSR·평균비용·TPS), 부트스트랩 CI(n=1,000, seed 고정),
  쌍별 McNemar, 효과크기(Cohen's d / Cramér's V), 파레토 프론티어.
- `pareto.png` — 정확도–비용 평면 시각화(파레토 프론티어 + S0–S1 보간선).

### 6.4 재채점(rejudge) — 부록 A.2 보정 재현

저장된 `raw_runs.jsonl` 을 **모델 재호출 없이** 보정 채점기로 다시 채점한다(토큰·비용·지연
불변, gold/pred/unit 만으로 재판정). 입력/출력 경로는 `src/rejudge.py` 상단 상수로 지정한다.

```bash
python -m src.rejudge       # results_finqa_format/ → results_finqa_format_rejudged/ 재채점
```

산출된 `REJUDGE_REPORT.md` 는 보정 전/후 TSR, 데이터셋·task_type별 변화, 잔존 실패 유형,
파레토 변화, 수동 검토 필요 목록(부호 불일치 등)을 정리한다.

---

## 7. 종속 변수·지표 (논문 표 1)

문항·반복·구조별 전수 로깅: query, final_answer, reasoning_trace, retrieved_evidence,
input/output tokens, latency, cost, 게이트 신호. 파생 지표:

- **TSR** (Task Success Rate): 보정 채점 정답 비율(run 단위 / 문항 단위 다수결).
- **Token**: 문항당 총 토큰(입력·출력 분리). usage_metadata 우선, tiktoken(cl100k_base) 교차검증.
- **Cost (USD)**: `config/pricing.yaml` 단가표로 산출.
- **TPS** (Tokens per Success): 성공 1건당 평균 토큰(정확도-비용 통합 효율; 낮을수록 효율적).
- **기권율**: 모델이 근거 부족으로 `INSUFFICIENT_EVIDENCE` 를 출력한 비율(오답과 분리 보고).

### 보정 채점기 (`src/metrics.py`)

선행 연구의 단순 일치 비교가 표현 형식 차이를 오답 처리하던 문제(예: `0.1954` ↔ `19.54%`)를
보정한다. 수치형은 단위 정규화(억원/백만원/천원/원/%/%pt/배/개, 천단위 구분기호·통화기호),
**퍼센트와 소수 비율의 100배 스케일 정합**(명시적 `%` 한쪽 한정, 같은 부호), 상대오차 < 1%
허용(절대·정수·극소값 특례, FinQA evaluation_schema 와 정합)을 적용한다. 텍스트형(항목명/
사업부문명)은 공백·괄호주석·대소문자 정규화 후 일치/포함. 부호만 다른 경우는 자동 정답
처리하지 않고 `judge_flag` 로만 표시해 저자 검토에 넘긴다. 단위 테스트: `pytest tests/`.

### 오류 유형(E1~E5) 분류

**저자 수기 분류**이며 자동화하지 않는다(논문 3.5 — 순환 타당성 회피). 실패 케이스는
`results/<sX>/failures.csv` 로 추출되고, `error_type`(저자 기입) 열과 보조 단서열
(`gold_chunk_retrieved`=E4 단서, `expr_recompute`=`used_expression` Python 재실행=E2 단서)을 포함한다.

---

## 8. 통계 분석 (`src/analysis.py`)

- TSR 차이: **McNemar 정확검정**(문항 단위 다수결 정답, 쌍체).
- 비용지표(log Token/Latency/TPS): **부트스트랩 CI n=1,000**(seed=20260601 고정).
- **within-run 순효과**: 동일 문항·구조 3회 반복 내 정답 전환 집계(비결정성 통제).
- 효과크기: Cohen's d(연속) / Cramér's V(범주). 파레토 프론티어 + S0–S1 보간선 시각화.

본문 H1·H2·H3(프레임워크 비교)는 이 저장소의 구조별 지표를 CrewAI 측 지표와 합쳐 판정한다(1절 주의 참고).

---

## 9. 저자 확정 프롬프트

`src/prompts/*.txt` 는 저자 확정본으로 **변경 금지**(변경 시 실험 통제 위반). 첫 줄의
`[[AUTHOR-LOCKED ...]]` 마커는 로더가 제거한다. 모든 노드는 system(역할/규칙/출력형식) +
user(INPUTS 슬롯)로 분리해 동일 프롬프트를 한국어·영어 문항에 공통 사용한다. JSON 출력은
파싱하며, 실패 시 동일 프롬프트로 1회 재호출 후에도 실패하면 그 run 을 실패로 기록한다.
근거 부족 시 모델은 `INSUFFICIENT_EVIDENCE` 로 기권하며 이는 오답(기권)으로 집계된다.

---

## 10. 라이선스·인용

- 코드: MIT License (`LICENSE`).
- K-DART-QA 데이터셋·구축 코드: Apache-2.0, <https://github.com/Asakhan/kdart_qa> (논문 §3.3.4).
- FinQA: 원 데이터셋 라이선스를 따른다(<https://github.com/czyssrs/FinQA>).
