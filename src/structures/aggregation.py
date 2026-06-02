"""S3 다수결 집계 (결정적 코드 — LLM 미사용).

알고리즘 (저자 명세):
 1) 각 후보 final_answer 정규화
 2) 수치 답은 상대오차 1% 이내를 같은 군집, 텍스트 답은 정규화 후 문자열 일치로 군집
 3) 최다 군집을 최종 답으로
 4) 3-way tie(셋 다 다른 군집)면 사전 우선순위(Executor=gemini_flash 후보) 채택
어느 후보가 채택됐는지 결과(aggregation_info)에 기록.
"""
from __future__ import annotations

from ..metrics import normalize_number, normalize_text, numeric_close


def aggregate_votes(candidates: list[dict], tiebreak_logical: str) -> tuple[dict, dict]:
    """후보 리스트 -> (chosen_candidate, aggregation_info)."""
    clusters: list[dict] = []
    for i, c in enumerate(candidates):
        ans = c.get("final_answer", "")
        num = normalize_number(ans)
        placed = False
        for cl in clusters:
            if cl["kind"] == "num" and num is not None and numeric_close(cl["rep"], num):
                cl["members"].append(i)
                placed = True
                break
            if cl["kind"] == "text" and num is None and normalize_text(ans) == cl["rep"]:
                cl["members"].append(i)
                placed = True
                break
        if not placed:
            if num is not None:
                clusters.append({"kind": "num", "rep": num, "members": [i]})
            else:
                clusters.append({"kind": "text", "rep": normalize_text(ans), "members": [i]})

    max_size = max((len(cl["members"]) for cl in clusters), default=0)
    top = [cl for cl in clusters if len(cl["members"]) == max_size]

    if max_size >= 2 and len(top) == 1:
        cluster = top[0]
        # 군집 내 대표: 우선순위 모델이 있으면 그것, 없으면 첫 멤버
        chosen_idx = _prefer(cluster["members"], candidates, tiebreak_logical)
        method = "majority"
    else:
        # 3-way tie 또는 모두 단일 군집 → 우선순위(Executor) 후보 채택
        chosen_idx = next(
            (i for i, c in enumerate(candidates) if c.get("logical_key") == tiebreak_logical),
            0 if candidates else -1,
        )
        method = "tiebreak"

    chosen = candidates[chosen_idx] if chosen_idx >= 0 else {"final_answer": ""}
    info = {
        "method": method,
        "n_candidates": len(candidates),
        "n_clusters": len(clusters),
        "max_cluster_size": max_size,
        "chosen_node": chosen.get("node"),
        "chosen_model": chosen.get("model"),
        "tiebreak_logical": tiebreak_logical,
        "candidates": [
            {"node": c.get("node"), "model": c.get("model"), "final_answer": c.get("final_answer")}
            for c in candidates
        ],
        "clusters": [
            {"kind": cl["kind"], "rep": cl["rep"], "size": len(cl["members"]), "members": cl["members"]}
            for cl in clusters
        ],
    }
    return chosen, info


def _prefer(member_indices: list[int], candidates: list[dict], tiebreak_logical: str) -> int:
    for idx in member_indices:
        if candidates[idx].get("logical_key") == tiebreak_logical:
            return idx
    return member_indices[0]
