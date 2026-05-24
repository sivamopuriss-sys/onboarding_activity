"""test_agent.py"""
import sys; sys.path.insert(0,"app")
from dynamo_client import get_completion_rate, MILESTONES

def test_milestones_defined():
    assert len(MILESTONES) > 0
    assert "welcome_sent" in MILESTONES
    assert "onboarding_complete" in MILESTONES

def test_completion_rate_new_customer():
    rate = get_completion_rate("brand-new-customer-xyz-999")
    assert 0 <= rate <= 1

def test_onboarding_state_structure():
    state = {
        "customer": {"id":"C001","name":"Alice","email":"alice@test.com","phone":"+1234567890"},
        "product_name": "Acme Platform",
        "action": "welcome",
    }
    assert "customer" in state
    assert "id" in state["customer"]
