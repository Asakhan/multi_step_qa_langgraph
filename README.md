# 다단계 수치 추론 문서 QA — 이종 LLM 오케스트레이션 구조(S0~S4) 정확도·비용 파레토 실험

석사논문 3장 실험 명세 구현. **LangGraph 단일 프레임워크**로 5개 오케스트레이션 구조
(S0~S4)를 동일 `AgentState`·동일 도구·동일 저자확정 프롬프트로 구현하고, S1~S4 본실험
(60문항 × 3반복 = 720회)을 백그라운드로 실행한 뒤 통계 분석까지 수행한다.

## 구조 요약

| 구조 | Executor | 추가 | 워크플로 | 가설 |
|---|---|---|---|---|
| S0 | gemini-2.5-flash | — | `retrieve→generate→END` 선형 (기준선) | — |
| S1 | gemini-3.1-pro | — | S0과 동일 그래프, generate 모델만 교체 | 상한 |
| S2 | gemini-2.5-flash | o3 검증 | `retrieve→generate→verify` 조건부, FAIL시 1회 재생성 | H1 |
| S3 | flash+gpt-4o-mini+claude-haiku-4.5 | 다수결 | `retrieve→[3 gen 병렬]→aggregate` | H1 |
| S4 | gemini-2.5-flash (3역할) | — | `decompose→solve→aggregate→END` (sub-q ≤5) | H2 |

구조 간 차이는 **노드·엣지 구성으로만** 발생한다(프롬프트·도구·상태 동일).

## 디렉토리

```
config/    pricing.yaml(단가) models.yaml(구조→모델 매핑) experiment.yaml(통제변수)
src/       state.py tools.py llm_clients.py metrics.py chunker.py datasets.py
           prompt_loader.py build_index.py runner.py orchestrator.py reporter.py
           notify.py analysis.py
           prompts/   ← 저자 확정 시스템 프롬프트(변경 금지)
           structures/  s0~s4.py + common.py(노드 빌더) + aggregation.py(S3 다수결)
data/      finqa/ kdart/ chroma/(인덱스)
results/   s0~s4/{raw_runs.jsonl,metrics.json,failures.csv}  analysis/{analysis.json,pareto.png}
logs/      백그라운드 로그 + run.pid
tests/     test_metrics.py
run.sh     진입점
```

## 데이터 출처 (이미 `data/` 에 확보 완료 — 자매 실험 multi_step_qa_langchain 과 동일)

- **FinQA (영어 20문항)**: `FinQA-main`(`data/finqa/FinQA-main`)의 test.json 에서
  `src/prepare_finqa.py` 가 canonical 20 id(단일근거 10[T3] + 다단계 10[T4])로 구성한
  `data/finqa/finqa.jsonl`. 자매 실험 finqa.jsonl 과 **id·정답·컨텍스트가 동일**함을 검증.
  각 문항은 self-contained `context_chunks`(text_N/table_N)를 포함 → **문항별 in-memory 코사인 검색**(전역 인덱스 불필요).
- **K-DART-QA (한국어 40문항)**: 자매 실험 `multi_step_qa_langchain/data/kdart_qa.jsonl`
  을 그대로 복사(`data/kdart/kdart_qa.jsonl`, gpt-4o-mini calibration 통과본, T1~T5).
  검색은 **자매 실험에서 구축한 전역 ChromaDB 인덱스**(`kdart_chunks`, **2,137 청크**)를
  `data/chroma/` 로 복사해 재사용 → 데이터셋 calibration 과 동일한 검색.

> FinQA·K-DART 모두 task_type 을 원본 T1~T5 그대로 사용한다(별도 매핑 없음).
> 검색기 시그니처는 동일(top_k=50, text-embedding-3-small, 코사인)하되 코퍼스 형태만
> 다르다: FinQA=문항별 컨텍스트, K-DART=전역 인덱스.

## 설치

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # Python 3.11/3.12, LangChain 0.3.x / LangGraph 0.2.x
cp .env.example .env                      # 그 후 .env 값을 채운다 (아래)
```

## .env 작성 (필수)

코드는 시작 시 `.env` 를 로드하고 필수 키가 비면 어떤 키가 빠졌는지 알려주며 즉시 중단한다.

| 키 | 용도 |
|---|---|
| `GOOGLE_API_KEY` | Gemini (flash/pro) |
| `OPENAI_API_KEY` | gpt-4o-mini, o3(경유 시), 임베딩 |
| `ANTHROPIC_API_KEY` | claude-haiku-4.5 |
| `MODEL_GEMINI_FLASH/PRO`, `MODEL_GPT4O_MINI`, `MODEL_CLAUDE_HAIKU`, `MODEL_O3` | 실제 API 모델 문자열 |
| `EMBEDDING_MODEL` | `text-embedding-3-small` |
| `CHROMA_DIR` | 인덱스 경로 (기본 `./data/chroma`) |
| `SMTP_HOST/PORT/USER/PASSWORD`, `MAIL_FROM`, `MAIL_TO` | 메일 보고 |
| `COST_CAP_USD` (선택) | 누적 비용 상한, 초과 시 일시정지+경고메일 |
| `SEND_EMAIL` (선택) | `0` 이면 메일 비활성화(테스트) |

> K-DART 검색 인덱스는 이미 구축된 것을 `data/chroma/` 로 복사해 재사용하므로 `DART_API_KEY` 는 불필요하다.

단가·모델매핑은 `config/*.yaml` 에서만 관리(코드 하드코딩 없음).

## 실행

```bash
# 1) 데이터 준비 + 코퍼스 검증 (FinQA finqa.jsonl 생성 + K-DART 인덱스 검증; 키 불필요)
./run.sh build-index

# 2) 표본 회귀검증 (구조별 데이터셋당 2문항 × 3반복, 전경)
./run.sh smoke

# 3) 본실험 — 백그라운드(nohup+setsid). 세션이 죽어도 계속 진행.
./run.sh all                  # S1~S4
./run.sh all --run-s0         # S0 포함(baseline 신규 실행)
./run.sh s2                   # 단일 구조
./run.sh s1 s3 s4             # 일부

# 4) 진행 확인/중단
./run.sh status               # PID + 로그 tail + 구조별 완료 수
./run.sh stop                 # 프로세스 그룹 종료

# 5) 진행 워처 (이 세션이 살아있는 동안에만 10분마다 진행 메일)
./run.sh watch --structures all

# 6) 통계 분석 + 최종 리포트(파레토 포함)
./run.sh analysis
```

- **체크포인트/재개**: `(구조,문항,반복)` 단위로 `raw_runs.jsonl` 에 append. 중단 후
  재실행하면 완료분은 건너뛰고 이어서 진행(idempotent). 실패(run_error) run 은 재시도.
- **구조 독립성**: orchestrator 가 각 구조를 별도 subprocess(`python -m src.runner`)로
  실행 → 한 구조 실패가 다른 구조에 영향 없음. 각 구조 완료 시 결과 저장 + 구조별 메일.
- **메일**: 구조별 완료 1통 + (워처가) 10분 진행 + 전체 완료 시 최종 종합(파레토 첨부).
  메일 실패는 로그만 남기고 실험을 중단하지 않는다.

## 검색 코퍼스 설계 (`config/experiment.yaml` `datasets.*.retrieval`)

| 데이터셋 | 코퍼스 | 검색 |
|---|---|---|
| **K-DART-QA** | 전역 ChromaDB 인덱스 `kdart_chunks` (2,137 청크, 자매 실험에서 구축 → `data/chroma/` 복사) | 쿼리 임베딩(text-embedding-3-small) → 코사인 top_k=50 → E4(검색실패) 측정 가능 |
| **FinQA** | 문항별 self-contained `context_chunks` (FinQA-main pre/post_text + table 행) | 문항 컨텍스트(25~38개) in-memory 코사인 top_k=50. top_k ≥ 컨텍스트 수이므로 전체 반환(임베딩 호출 없음) |

> K-DART 인덱스는 **이미 구축된 것을 복사**하므로 재색인이 필요 없다(`./run.sh build-index` 는
> finqa.jsonl 생성 + 인덱스 존재·개수 검증만 수행). FinQA 를 FinQA-main 에서 다시 만들려면
> `python -m src.prepare_finqa`.

## 종속변수·지표

문항·반복·구조별 전수 로깅(query, final_answer, reasoning_trace, retrieved_evidence,
input/output tokens, latency) + 파생지표: **TSR**(EM+수치 상대오차<1%), **Latency**,
**Token**(입력·출력 분리), **Cost**(USD), **TPS**(성공당 토큰).

정답 판정(`src/metrics.py`): 수치형은 단위 표기 차이 인정(억원/백만원/천원/원/%/%pt/배/개,
천단위 구분기호, 통화기호 정규화 후 상대오차<1%, 정수·극소값 특례 — FinQA evaluation_schema
와 정합). 텍스트형(항목명/사업부문명)은 공백·괄호주석·대소문자 정규화 후 일치/포함.
경계 케이스는 `judge_flag` 로 표시. 단위 테스트: `pytest tests/`.

## 통계 분석 (`src/analysis.py`)

- TSR 차이: **McNemar 정확검정**(문항 단위 다수결 정답, 쌍체)
- 비용지표(log Token/Latency/TPS): **부트스트랩 CI n=1,000** (seed 고정)
- **within-run 순효과(net flip)**: 동일 문항·구조 3회 반복 내 정답 전환 집계(비결정성 통제)
- **H1**: S2·S3 TPS 를 S0 와 비교 — 적어도 하나가 유의하게 낮으면 지지(TPS차 부트스트랩 CI 상한<0)
- **H2**: S4 vs S0 TSR McNemar + 과업유형(T1~T5, 특히 **T4**) 분해
- 효과크기: Cohen's d(연속) / Cramér's V(범주), 파레토 프론티어 + S0–S1 보간선 시각화

> **오류유형(E1~E5) 분류는 저자 수기 작업** — 자동화하지 않는다. 실패 케이스는
> `results/sX/failures.csv` 로 추출되며 `error_type`(저자 기입) 열과 보조열
> (`gold_chunk_retrieved`=E4 단서, `expr_recompute`=산술식 Python 재실행=E2 단서)을 포함.

## 저자 확정 프롬프트

`src/prompts/*.txt` 는 저자 확정본으로 **변경 금지**(변경 시 실험 통제 위반). 각 파일 첫 줄의
`[[AUTHOR-LOCKED ...]]` 마커는 로더가 제거하며 프롬프트 본문에는 포함되지 않는다. 모든 노드는
system(역할/규칙/출력형식) + user(INPUTS 슬롯)로 분리해 동일 프롬프트를 한국어·영어 문항에
공통 사용한다. JSON 출력은 파싱하며, 실패 시 동일 프롬프트로 1회 재호출 후에도 실패하면 그
run 을 실패로 기록한다.
