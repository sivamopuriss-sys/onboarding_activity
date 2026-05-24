"""agent.py — LangGraph onboarding agent"""
from typing import TypedDict, Optional, Annotated
import operator
from langgraph.graph import StateGraph, END
from nodes import (node_welcome, node_answer_question, node_schedule_call,
                   node_check_progress, node_send_nudge)


class OnboardingState(TypedDict):
    customer: dict                       # {id, name, email, phone, plan}
    product_name: str
    customer_question: Optional[str]
    action: str                          # welcome | question | schedule | progress
    welcome_result: Optional[dict]
    rag_answer: Optional[str]
    sources: list[str]
    calendar_result: Optional[dict]
    completion_pct: Optional[float]
    needs_nudge: bool
    nudge_result: Optional[dict]
    milestone: Optional[str]
    warnings: Annotated[list, operator.add]


def route_action(state: OnboardingState) -> str:
    action = state.get("action","welcome")
    return {
        "welcome":  "welcome",
        "question": "answer_question",
        "schedule": "schedule_call",
        "progress": "check_progress",
    }.get(action, "welcome")


def build_onboarding_agent():
    g = StateGraph(OnboardingState)
    g.add_node("welcome",         node_welcome)
    g.add_node("answer_question", node_answer_question)
    g.add_node("schedule_call",   node_schedule_call)
    g.add_node("check_progress",  node_check_progress)
    g.add_node("send_nudge",      node_send_nudge)

    g.set_conditional_entry_point(route_action, {
        "welcome":         "welcome",
        "answer_question": "answer_question",
        "schedule_call":   "schedule_call",
        "check_progress":  "check_progress",
    })

    g.add_edge("welcome",         END)
    g.add_edge("answer_question", END)
    g.add_edge("schedule_call",   END)
    g.add_edge("check_progress",  "send_nudge")
    g.add_edge("send_nudge",      END)

    return g.compile()

agent = build_onboarding_agent()


def run_onboarding(customer: dict, action: str = "welcome",
                   question: str = None, product_name: str = "our platform") -> dict:
    result = agent.invoke({
        "customer":          customer,
        "product_name":      product_name,
        "customer_question": question,
        "action":            action,
        "welcome_result":    None,
        "rag_answer":        None,
        "sources":           [],
        "calendar_result":   None,
        "completion_pct":    None,
        "needs_nudge":       False,
        "nudge_result":      None,
        "milestone":         None,
        "warnings":          [],
    })
    return {k: v for k, v in result.items() if v is not None}
