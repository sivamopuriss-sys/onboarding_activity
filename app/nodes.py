"""nodes.py — LangGraph onboarding agent node implementations"""
from langsmith_config import setup_langsmith
from rag import get_product_answer
from twilio_client import send_welcome_sms, send_checkin_sms
from calendar_client import schedule_onboarding_call
from dynamo_client import update_milestone, get_completion_rate
from guardrails.safety import check_input

setup_langsmith("project-30-onboarding-agent")


def node_welcome(state: dict) -> dict:
    """Send welcome SMS and log milestone."""
    customer = state["customer"]
    # Guardrails: check phone number format (basic PII validation)
    clean_phone, warnings = check_input(customer.get("phone",""))

    result = send_welcome_sms(
        to_number=customer.get("phone",""),
        customer_name=customer.get("name","Customer"),
        product_name=state.get("product_name","our platform"),
    )
    update_milestone(customer["id"], "welcome_sent")
    return {"welcome_result": result, "warnings": warnings, "milestone": "welcome_sent"}


def node_answer_question(state: dict) -> dict:
    """Answer a product question via RAG knowledge base."""
    question = state.get("customer_question","")
    if not question:
        return {"rag_answer":"No question provided.","sources":[]}
    result = get_product_answer(question)
    return {"rag_answer": result["answer"], "sources": result["sources"]}


def node_schedule_call(state: dict) -> dict:
    """Schedule an onboarding check-in call."""
    customer = state["customer"]
    result   = schedule_onboarding_call(
        customer_name=customer.get("name","Customer"),
        customer_email=customer.get("email",""),
        days_from_now=7,
    )
    update_milestone(customer["id"], "check_in_1_sent")
    return {"calendar_result": result, "milestone": "check_in_1_sent"}


def node_check_progress(state: dict) -> dict:
    """Check onboarding completion rate and decide next action."""
    customer_id    = state["customer"]["id"]
    completion_pct = get_completion_rate(customer_id)
    needs_nudge    = completion_pct < 0.5
    return {"completion_pct": completion_pct, "needs_nudge": needs_nudge}


def node_send_nudge(state: dict) -> dict:
    """Send a check-in SMS to customers who are falling behind."""
    customer = state["customer"]
    if state.get("needs_nudge"):
        result = send_checkin_sms(
            to_number=customer.get("phone",""),
            customer_name=customer.get("name","Customer"),
            checkin_number=1,
        )
        update_milestone(customer["id"], "check_in_2_sent")
        return {"nudge_result": result, "milestone": "check_in_2_sent"}
    return {"nudge_result": None}
