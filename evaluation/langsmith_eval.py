"""langsmith_eval.py — Onboarding quality evaluation"""
from langsmith import Client
client = Client()

def milestone_completion_evaluator(run, example):
    """Check if the correct milestone was logged."""
    milestone = run.outputs.get("milestone","")
    expected  = example.outputs.get("expected_milestone","")
    correct   = milestone == expected if expected else True
    return {"key":"milestone_accuracy","score":1.0 if correct else 0.0}

def response_quality_evaluator(run, example):
    """Check if RAG answer contains key information."""
    answer   = str(run.outputs.get("rag_answer",""))
    keywords = example.outputs.get("expected_keywords",[])
    if not keywords: return {"key":"response_quality","score":1.0}
    hits  = sum(1 for kw in keywords if kw.lower() in answer.lower())
    score = hits / len(keywords)
    return {"key":"response_quality","score":score}

print("LangSmith evaluators ready. Use with evaluate() to run evals.")
