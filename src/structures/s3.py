"""S3 (다수결): retrieve -> [gen_gemini, gen_gpt, gen_claude] 병렬 fan-out -> aggregate fan-in.
각 노드는 candidate_answers 에 누적. 집계는 결정적 다수결(aggregation.aggregate_votes).
"""
from __future__ import annotations

import time

from langgraph.graph import StateGraph, START, END

from .. import config
from ..state import AgentState
from .common import make_retrieve_node, make_vote_gen_node
from .aggregation import aggregate_votes

_TIEBREAK = config.models_cfg()["s3_tiebreak_priority"]


def make_aggregate_node():
    def aggregate_node(state):
        t0 = time.perf_counter()
        candidates = state.get("candidate_answers", [])
        chosen, info = aggregate_votes(candidates, _TIEBREAK)
        return {
            "final_answer": chosen.get("final_answer", ""),
            "reasoning_trace": chosen.get("reasoning", ""),
            "used_expression": list(chosen.get("expressions", []) or []),
            "aggregation_info": info,
            # 집계는 LLM 미사용 → token_usage 없음
            "node_latencies": [{"node": "aggregate", "seconds": time.perf_counter() - t0}],
        }
    return aggregate_node


def build():
    g = StateGraph(AgentState)
    g.add_node("retrieve", make_retrieve_node())
    g.add_node("gen_gemini", make_vote_gen_node("s3", "gen_gemini", "gen_gemini"))
    g.add_node("gen_gpt", make_vote_gen_node("s3", "gen_gpt", "gen_gpt"))
    g.add_node("gen_claude", make_vote_gen_node("s3", "gen_claude", "gen_claude"))
    g.add_node("aggregate", make_aggregate_node())

    g.add_edge(START, "retrieve")
    # fan-out (병렬)
    g.add_edge("retrieve", "gen_gemini")
    g.add_edge("retrieve", "gen_gpt")
    g.add_edge("retrieve", "gen_claude")
    # fan-in: 세 노드 모두 끝난 뒤 aggregate 실행
    g.add_edge("gen_gemini", "aggregate")
    g.add_edge("gen_gpt", "aggregate")
    g.add_edge("gen_claude", "aggregate")
    g.add_edge("aggregate", END)
    return g.compile()
