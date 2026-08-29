"""백그라운드 실험 본체. 구조들을 독립 프로세스로 순차 실행 후 최종 분석·메일.

  python -m src.orchestrator --structures s1 s2 s3 s4 [--run-s0]
                             [--runs 3] [--datasets finqa kdart]
                             [--limit-per-dataset N] [--no-final-analysis]

- 각 구조를 별도 subprocess(python -m src.runner)로 실행 → 한 구조 실패가 다른 구조에
  영향을 주지 않음. 각 구조 완료 시 runner 가 결과 저장 + 구조별 메일 발송.
- 전 구조 완료 후 통계 분석(analysis) 실행 + 최종 종합 리포트 메일(파레토 첨부).
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import traceback
from datetime import datetime, timezone

from . import config


class _Terminated(Exception):
    """SIGTERM/SIGINT 수신을 예외로 승격해 finally 정리·중단메일을 보장."""


def _progress_summary(structures: list[str]) -> str:
    """현재까지 구조별 완료 run 수·성공·비용 요약(중단/종료 메일 본문용)."""
    lines = []
    for s in structures:
        path = config.RESULTS_DIR / s / "raw_runs.jsonl"
        total = success = error = 0
        cost = 0.0
        if path.exists():
            for line in open(path, encoding="utf-8"):
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                total += 1
                if r.get("run_error"):
                    error += 1
                else:
                    success += int(r.get("success", 0))
                    cost += float(r.get("cost", 0.0))
        lines.append(f"  {s.upper()}: {total} runs (성공 {success}, 오류 {error}, 비용 ${cost:.3f})")
    return "\n".join(lines) if lines else "  (진행 데이터 없음)"


def _safe_mail(subject: str, body: str) -> None:
    """메일 발송(실패가 종료를 막지 않도록 전부 흡수)."""
    try:
        from . import notify
        notify.send_email(subject, body)
    except Exception as exc:  # noqa: BLE001
        print(f"[orchestrator] 메일 발송 실패: {exc}", flush=True)


def _expand(structures: list[str], run_s0: bool) -> list[str]:
    out = []
    for s in structures:
        if s == "all":
            out.extend(["s1", "s2", "s3", "s4"])
        else:
            out.append(s)
    if run_s0 and "s0" not in out:
        out = ["s0"] + out
    # 중복 제거(순서 보존)
    seen = set()
    ordered = []
    for s in out:
        if s not in seen:
            seen.add(s)
            ordered.append(s)
    return ordered


def main(argv=None):
    ap = argparse.ArgumentParser(description="실험 오케스트레이터 (백그라운드 본체)")
    ap.add_argument("--structures", nargs="+", required=True)
    ap.add_argument("--run-s0", action="store_true")
    ap.add_argument("--runs", type=int, default=config.experiment_cfg()["repetition"]["runs_per_question"])
    ap.add_argument("--datasets", nargs="+", default=["finqa", "kdart"],
                    choices=["finqa", "finqa_ext", "kdart"])
    ap.add_argument("--limit-per-dataset", type=int, default=None)
    ap.add_argument("--no-final-analysis", action="store_true")
    args = ap.parse_args(argv)

    structures = _expand(args.structures, args.run_s0)
    print(f"[orchestrator] {datetime.now(timezone.utc).isoformat()} 구조: {structures}", flush=True)

    # SIGTERM/SIGINT 를 예외로 승격 → 신호 종료(./run.sh stop, 세션 종료) 시에도 중단메일 후 정리.
    def _sig_handler(signum, _frame):
        raise _Terminated(f"signal {signum}")
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except (ValueError, OSError):
        pass  # 비메인 스레드 등에서 핸들러 등록 불가 시 무시

    # 전체 필요한 env 사전 검증(빠른 실패). self-contained 컨텍스트라 임베딩 불요.
    config.validate_env(structures, need_email=True, need_embedding=False)

    results: dict[str, int] = {}
    notified = {"done": False}  # 정상 최종리포트/중단메일 발송 시 finally 중복 방지

    try:
        for s in structures:
            cmd = [sys.executable, "-m", "src.runner", "--structure", s,
                   "--runs", str(args.runs), "--datasets", *args.datasets]
            if args.limit_per_dataset:
                cmd += ["--limit-per-dataset", str(args.limit_per_dataset)]
            print(f"\n[orchestrator] === {s.upper()} 시작: {' '.join(cmd)} ===", flush=True)
            try:
                proc = subprocess.run(cmd, cwd=str(config.ROOT))
                results[s] = proc.returncode
                print(f"[orchestrator] {s.upper()} 종료 코드 {proc.returncode}", flush=True)
            except _Terminated:
                raise  # 신호는 상위 except 로 전파
            except Exception as exc:  # noqa: BLE001 — 한 구조 실패가 다른 구조를 막지 않음
                results[s] = -1
                print(f"[orchestrator] {s.upper()} 예외: {exc}", flush=True)

        print(f"\n[orchestrator] 구조별 종료코드: {results}", flush=True)

        if not args.no_final_analysis:
            try:
                from . import analysis, notify
                print("[orchestrator] 최종 통계 분석 실행", flush=True)
                res = analysis.run_full_analysis()
                body = analysis.final_report_text(res)
                print(body, flush=True)
                attachments = [res["pareto"]["plot"]] if res.get("pareto", {}).get("plot") else []
                notify.send_final_report(body, attachments)
                notified["done"] = True  # 정상 최종리포트 발송 → finally 중복 방지
            except Exception as exc:  # noqa: BLE001
                print(f"[orchestrator] 최종 분석/메일 실패: {exc}", flush=True)

        print("[orchestrator] 완료", flush=True)
        return results

    except BaseException as exc:  # noqa: BLE001 — 비정상/신호 종료 시 중단메일 보장
        kind = "신호 종료" if isinstance(exc, (_Terminated, KeyboardInterrupt)) else "예외"
        tb = "" if isinstance(exc, (_Terminated, KeyboardInterrupt)) else traceback.format_exc()[-2000:]
        print(f"[orchestrator] 비정상 종료({kind}): {exc}", flush=True)
        _safe_mail(
            f"[논문실험][중단] 비정상 종료 ({kind})",
            f"오케스트레이터가 비정상 종료되었습니다.\n원인: {type(exc).__name__}: {exc}\n\n"
            f"[진행 요약]\n{_progress_summary(structures)}\n\n{tb}")
        notified["done"] = True
        raise
    finally:
        # 정상 최종리포트도, 중단메일도 안 나간 경로(예: --no-final-analysis 정상 종료)에서
        # 종료 요약 메일 1회 발송.
        if not notified["done"]:
            _safe_mail("[논문실험][종료] 실행 종료 요약",
                       f"오케스트레이터 실행이 종료되었습니다.\n\n[진행 요약]\n{_progress_summary(structures)}")
            notified["done"] = True


if __name__ == "__main__":
    main()
