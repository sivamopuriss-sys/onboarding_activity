"""dynamo_client.py — DynamoDB milestone tracking"""
import os, boto3
from datetime import datetime
from dotenv import load_dotenv
load_dotenv()

TABLE = os.getenv("DYNAMODB_TABLE","customer-onboarding")
MILESTONES = [
    "welcome_sent","product_intro_completed","first_login",
    "setup_completed","check_in_1_sent","check_in_2_sent","onboarding_complete"
]


def get_client():
    return boto3.resource("dynamodb", region_name=os.getenv("AWS_REGION","us-east-1"))


def get_customer_state(customer_id: str) -> dict:
    """Get current onboarding state for a customer."""
    try:
        table = get_client().Table(TABLE)
        resp  = table.get_item(Key={"customer_id": customer_id})
        return resp.get("Item", {"customer_id":customer_id,"milestones":{},"messages":[]})
    except Exception:
        return {"customer_id":customer_id,"milestones":{},"messages":[]}


def update_milestone(customer_id: str, milestone: str) -> bool:
    """Mark a milestone as completed."""
    if milestone not in MILESTONES:
        return False
    try:
        table = get_client().Table(TABLE)
        table.update_item(
            Key={"customer_id": customer_id},
            UpdateExpression="SET milestones.#m = :ts",
            ExpressionAttributeNames={"#m": milestone},
            ExpressionAttributeValues={":ts": datetime.utcnow().isoformat()},
        )
        return True
    except Exception as e:
        print(f"DynamoDB update failed: {e}")
        return False


def get_completion_rate(customer_id: str) -> float:
    """Calculate onboarding completion percentage."""
    state = get_customer_state(customer_id)
    completed = len(state.get("milestones",{}))
    return completed / len(MILESTONES)
