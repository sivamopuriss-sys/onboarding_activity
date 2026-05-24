"""api.py — FastAPI + Lambda handler via Mangum"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from mangum import Mangum
from agent import run_onboarding

app = FastAPI(title="Intelligent Customer Onboarding Agent")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class OnboardingRequest(BaseModel):
    customer: dict                  # {id, name, email, phone, plan}
    action: str = "welcome"        # welcome | question | schedule | progress
    question: Optional[str] = None
    product_name: str = "Acme Platform"

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/onboard")
def onboard(req: OnboardingRequest):
    if not req.customer.get("id"):
        raise HTTPException(422,"Customer ID required.")
    return run_onboarding(req.customer, req.action, req.question, req.product_name)

# AWS Lambda handler
handler = Mangum(app)
