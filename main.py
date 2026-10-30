
import os
import logging

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI()

# Groq API key
api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError("GROQ_API_KEY is missing!")

# Initialize Groq model
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    api_key=api_key
)

# Request validation
class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)

# Health check
@app.get("/health")
def health():
    logger.info("Health check called")
    return {"status": "API is running"}


import time

@app.post("/chat")
def chat(request: ChatRequest):

    start_time = time.time()

    try:
        response = llm.invoke(request.query)

        end_time = time.time()
        response_time = round(end_time - start_time, 2)

        logger.info(f"Response time: {response_time} seconds")

        return {
            "answer": response.content,
            "response_time": response_time
        }

    except Exception:
        logger.exception("Chat error")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong"
        )
