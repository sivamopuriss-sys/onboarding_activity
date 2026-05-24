FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir langgraph langchain langchain-openai langchain-pinecone pinecone-client openai twilio google-api-python-client google-auth boto3 fastapi uvicorn mangum pydantic langsmith python-dotenv tiktoken
COPY app/ ./app/
COPY guardrails/ ./guardrails/
COPY langsmith_config.py .
EXPOSE 8080
CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "8080"]
