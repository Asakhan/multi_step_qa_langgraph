"""메일 발송 (env 기반 SMTP). 메일 실패는 로그만 남기고 예외를 올리지 않는다.

연결된 Gmail MCP 가 있으면 그 경로도 옵션이나(에이전트가 호출), 파이썬 본체의 기본
채널은 SMTP 이다. SEND_EMAIL=0 으로 비활성화 가능(테스트용).
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path


def _enabled() -> bool:
    return os.environ.get("SEND_EMAIL", "1").strip() not in ("0", "false", "False", "")


def send_email(subject: str, body: str, attachments: list[str] | None = None) -> bool:
    if not _enabled():
        print(f"  [mail-disabled] {subject}")
        return False
    host = os.environ.get("SMTP_HOST", "").strip()
    port = int(os.environ.get("SMTP_PORT", "587") or "587")
    user = os.environ.get("SMTP_USER", "").strip()
    pwd = os.environ.get("SMTP_PASSWORD", "").strip()
    mail_from = os.environ.get("MAIL_FROM", user).strip()
    mail_to = os.environ.get("MAIL_TO", "").strip()
    if not (host and mail_to):
        print(f"  [mail-skip] SMTP 미설정 — '{subject}' 미발송")
        return False

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = mail_from
    msg["To"] = mail_to
    msg.set_content(body)
    for path in attachments or []:
        p = Path(path)
        if not p.exists():
            continue
        data = p.read_bytes()
        sub = "octet-stream"
        maintype = "application"
        if p.suffix.lower() in (".png", ".jpg", ".jpeg"):
            maintype, sub = "image", p.suffix.lower().lstrip(".")
        msg.add_attachment(data, maintype=maintype, subtype=sub, filename=p.name)

    try:
        ctx = ssl.create_default_context()
        if port == 465:
            with smtplib.SMTP_SSL(host, port, context=ctx, timeout=60) as s:
                if user:
                    s.login(user, pwd)
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=60) as s:
                s.starttls(context=ctx)
                if user:
                    s.login(user, pwd)
                s.send_message(msg)
        print(f"  [mail-sent] {subject} -> {mail_to}")
        return True
    except Exception as exc:  # noqa: BLE001 — 메일 실패가 실험을 막지 않음
        print(f"  [mail-error] '{subject}': {exc}")
        return False


def _fmt(x, nd=4):
    if x is None:
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def send_structure_report(structure: str, summary: dict) -> bool:
    tsr = summary.get("TSR")
    tps = summary.get("TPS")
    subj = (f"[논문실험] {structure.upper()} 완료 — "
            f"TSR {_fmt(tsr,3)}, TPS {_fmt(tps,0) if tps else 'N/A'}")
    lines = [
        f"구조: {structure.upper()}",
        f"총 run: {summary.get('n_runs')} (오류 {summary.get('n_error_runs',0)}, "
        f"검토플래그 {summary.get('n_judge_flagged',0)})",
        f"TSR: {_fmt(tsr,4)}",
        f"평균 입력토큰: {_fmt(summary.get('avg_input_tokens'),1)}",
        f"평균 출력토큰: {_fmt(summary.get('avg_output_tokens'),1)}",
        f"평균 총토큰: {_fmt(summary.get('avg_total_tokens'),1)}",
        f"평균 지연(초): {_fmt(summary.get('avg_latency'),2)}",
        f"총비용(USD): {_fmt(summary.get('total_cost_usd'),4)}",
        f"TPS (성공당 토큰): {_fmt(tps,1) if tps else 'N/A'}",
        f"성공당 비용(USD): {_fmt(summary.get('cost_per_success_usd'),5)}",
        "",
        "[데이터셋별 TSR]",
    ]
    for ds, m in (summary.get("by_dataset") or {}).items():
        lines.append(f"  - {ds}: TSR {_fmt(m['TSR'],3)} "
                     f"(성공 {m['n_success']}/{m['n_runs']}), 평균토큰 {_fmt(m['avg_tokens'],0)}")
    lines.append("")
    lines.append("[과업유형별 TSR]")
    for tt, m in (summary.get("by_task_type") or {}).items():
        lines.append(f"  - {tt}: TSR {_fmt(m['TSR'],3)} (성공 {m['n_success']}/{m['n_runs']})")
    lines.append("")
    lines.append(f"결과 파일: results/{structure}/raw_runs.jsonl, metrics.json, failures.csv")
    return send_email(subj, "\n".join(lines))


def send_cost_warning(structure: str, cum: float, cap: float) -> bool:
    return send_email(
        f"[논문실험][경고] 비용 상한 도달 — {structure.upper()} 일시정지",
        f"누적 비용 ${cum:.2f} 가 상한 COST_CAP_USD=${cap:.2f} 에 도달하여 "
        f"{structure.upper()} 실행을 일시정지했습니다.\n"
        f"상한을 올리거나 해제 후 재실행하면 이어서 진행됩니다(idempotent resume).")


def send_progress(body: str, subject: str | None = None) -> bool:
    return send_email(subject or "[논문실험] 진행 보고", body)


def send_final_report(body: str, attachments: list[str] | None = None) -> bool:
    return send_email("[논문실험] 최종 종합 리포트 (H1·H2 검정 + 파레토)", body, attachments)
