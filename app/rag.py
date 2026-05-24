"""rag.py — Pinecone RAG for product Q&A"""
import os
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()

INDEX_NAME = os.getenv("PINECONE_INDEX","onboarding-kb")

PROMPT = """You are a friendly, helpful customer onboarding assistant.
Answer the customer's question using the product knowledge base below.
Be concise, clear, and encouraging. If the answer is not in the KB, say so and offer to escalate.

Knowledge Base:
{context}

Customer Question: {question}

Answer:"""

def get_product_answer(question: str) -> dict:
    embeddings  = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = PineconeVectorStore(index_name=INDEX_NAME, embedding=embeddings)
    retriever   = vectorstore.as_retriever(search_kwargs={"k":3})
    llm         = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
    prompt      = PromptTemplate(template=PROMPT, input_variables=["context","question"])
    chain       = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=retriever,
        return_source_documents=True, chain_type_kwargs={"prompt":prompt})
    result  = chain.invoke({"query":question})
    sources = [d.metadata.get("source","KB") for d in result.get("source_documents",[])]
    return {"answer":result["result"],"sources":sources}
