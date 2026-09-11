from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

from src.agent import CustomerSupportAgent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
)


# --------------------------------------------------
# Application lifespan
# --------------------------------------------------

@asynccontextmanager
async def lifespan(
    app: FastAPI,
):
    """
    Load the AI model once when the API starts.

    We do NOT want to reload the model
    for every request.
    """

    print("=" * 60)
    print("STARTING ARABIC TOOL-CALLING API")
    print("=" * 60)

    # Load model + LoRA + pipeline + router once.
    app.state.agent = (
        CustomerSupportAgent()
    )

    print("\n✅ API startup completed.")

    yield

    print("\nShutting down API...")


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="Arabic-English Tool-Calling LLM API",
    description=(
        "Fine-tuned bilingual LLM for "
        "customer-support tool calling."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------
# Root endpoint
# --------------------------------------------------

@app.get("/")
def root() -> dict:

    return {
        "name": (
            "Arabic-English "
            "Tool-Calling LLM API"
        ),
        "version": "1.0.0",
        "docs": "/docs",
    }


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/health")
def health(
    request: Request,
) -> dict:

    model_loaded = hasattr(
        request.app.state,
        "agent",
    )

    return {
        "status": "healthy",
        "model_loaded": model_loaded,
    }


# --------------------------------------------------
# Chat endpoint
# --------------------------------------------------

@app.post(
    "/chat",
    response_model=ChatResponse,
)
def chat(
    payload: ChatRequest,
    request: Request,
) -> ChatResponse:
    """
    Run the complete AI agent:

    User message
        ↓
    LLM
        ↓
    Tool call
        ↓
    Router
        ↓
    Tool execution
    """

    agent = request.app.state.agent

    result = agent.run(
        payload.message
    )

    return ChatResponse(
        **result
    )