"""calendar_client.py — Google Calendar scheduling"""
import os
from datetime import datetime, timedelta
from googleapiclient.discovery import build
from google.oauth2 import service_account
from dotenv import load_dotenv
load_dotenv()

CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID","primary")
SCOPES      = ["https://www.googleapis.com/auth/calendar"]


def get_calendar_service():
    """Get Google Calendar service. Uses service account credentials."""
    creds_file = os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE","")
    if not creds_file or not os.path.exists(creds_file):
        print("[DEMO] Google Calendar not configured — returning mock service")
        return None
    creds = service_account.Credentials.from_service_account_file(creds_file, scopes=SCOPES)
    return build("calendar", "v3", credentials=creds)


def schedule_onboarding_call(customer_name: str, customer_email: str,
                              days_from_now: int = 7) -> dict:
    """Schedule a 30-minute onboarding call."""
    service = get_calendar_service()
    if not service:
        return {"status":"demo","message":f"Would schedule call for {customer_name} in {days_from_now} days"}

    start = datetime.utcnow() + timedelta(days=days_from_now)
    start = start.replace(hour=14, minute=0, second=0, microsecond=0)
    end   = start + timedelta(minutes=30)

    event = {
        "summary": f"Onboarding Call — {customer_name}",
        "description": "30-minute onboarding success call to ensure smooth setup.",
        "start": {"dateTime": start.isoformat()+"Z", "timeZone": "UTC"},
        "end":   {"dateTime": end.isoformat()+"Z",   "timeZone": "UTC"},
        "attendees": [{"email": customer_email}],
        "conferenceData": {"createRequest": {"requestId": f"onboard-{customer_name}"}},
    }
    result = service.events().insert(calendarId=CALENDAR_ID, body=event,
                                      conferenceDataVersion=1).execute()
    return {"status":"scheduled","event_id":result.get("id"),"link":result.get("htmlLink","")}
