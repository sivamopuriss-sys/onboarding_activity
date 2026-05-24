"""twilio_client.py — Twilio SMS integration"""
import os
from twilio.rest import Client
from dotenv import load_dotenv
load_dotenv()

def get_client():
    return Client(os.getenv("TWILIO_ACCOUNT_SID"), os.getenv("TWILIO_AUTH_TOKEN"))

def send_sms(to_number: str, message: str) -> dict:
    """Send an SMS via Twilio. Returns message SID or error."""
    from_number = os.getenv("TWILIO_FROM_NUMBER")
    if not from_number or not os.getenv("TWILIO_ACCOUNT_SID"):
        print(f"[DEMO] Would send SMS to {to_number}: {message[:60]}...")
        return {"sid": "DEMO_SID", "status": "demo"}
    try:
        client = get_client()
        msg    = client.messages.create(body=message, from_=from_number, to=to_number)
        return {"sid": msg.sid, "status": msg.status}
    except Exception as e:
        return {"error": str(e), "status": "failed"}

def send_welcome_sms(to_number: str, customer_name: str, product_name: str) -> dict:
    message = (
        f"Hi {customer_name}! Welcome to {product_name}. "
        f"Your account is ready. Reply with any questions and our AI assistant will help. "
        f"Let's get you set up! 🚀"
    )
    return send_sms(to_number, message)

def send_checkin_sms(to_number: str, customer_name: str, checkin_number: int) -> dict:
    messages = {
        1: f"Hi {customer_name}, it's been 3 days since you joined! Have you completed setup? Reply YES or I need help.",
        2: f"Hey {customer_name}! Quick check-in — how are you getting on? Any questions? We're here to help.",
    }
    return send_sms(to_number, messages.get(checkin_number, messages[1]))
