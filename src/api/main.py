from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import (
    FastAPI,
    HTTPException,
    Request,
)

from src.agent import CustomerSupportAgent
from src.api.schemas import (
    ChatRequest,
    ChatResponse,
)
from src.core.logging_config import logger


# --------------------------------------------------
# Application factory
# --------------------------------------------------

def create_app(
    agent_factory=CustomerSupportAgent,
) -> FastAPI:
    """
    Create the FastAPI application.

    agent_factory allows tests to inject
    a fake agent without loading the LLM.
    """

    # --------------------------------------------------
    # Lifespan
    # --------------------------------------------------

    @asynccontextmanager
    async def lifespan(
        app: FastAPI,
    ):

        logger.info(
            "Starting Arabic tool-calling API"
        )

        # Load model once during production startup.
        app.state.agent = agent_factory()

        logger.info(
            "AI agent loaded successfully"
        )

        yield

        logger.info(
            "Shutting down API"
        )

    # --------------------------------------------------
    # Application
    # --------------------------------------------------

    app = FastAPI(
        title=(
            "Arabic-English "
            "Tool-Calling LLM API"
        ),
        description=(
            "Fine-tuned bilingual LLM for "
            "customer-support tool calling."
        ),
        version="1.0.0",
        lifespan=lifespan,
    )

    # --------------------------------------------------
    # Request logging middleware
    # --------------------------------------------------

    @app.middleware("http")
    async def request_logging(
        request: Request,
        call_next,
    ):

        request_id = (
            request.headers.get(
                "X-Request-ID"
            )
            or str(uuid4())
        )

        request.state.request_id = (
            request_id
        )

        logger.info(
            "request_started "
            "request_id=%s method=%s path=%s",
            request_id,
            request.method,
            request.url.path,
        )

        try:

            response = await call_next(
                request
            )

        except Exception:

            logger.exception(
                "request_failed "
                "request_id=%s",
                request_id,
            )

            raise

        response.headers[
            "X-Request-ID"
        ] = request_id

        logger.info(
            "request_completed "
            "request_id=%s status=%s",
            request_id,
            response.status_code,
        )

        return response

    # --------------------------------------------------
    # Root
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
    # Health
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
            "status": (
                "healthy"
                if model_loaded
                else "unhealthy"
            ),
            "model_loaded": (
                model_loaded
            ),
        }

    # --------------------------------------------------
    # Chat
    # --------------------------------------------------

    @app.post(
        "/chat",
        response_model=ChatResponse,
    )
    def chat(
        payload: ChatRequest,
        request: Request,
    ) -> ChatResponse:

        agent = request.app.state.agent

        request_id = (
            request.state.request_id
        )

        logger.info(
            "agent_request "
            "request_id=%s",
            request_id,
        )

        try:

            result = agent.run(
                payload.message
            )

        # Known model / validation problems.
        except (
            ValueError,
            TypeError,
        ) as error:

            logger.warning(
                "agent_validation_error "
                "request_id=%s error=%s",
                request_id,
                error,
            )

            raise HTTPException(
                status_code=422,
                detail=(
                    "The AI model produced "
                    "an invalid tool call."
                ),
            ) from error

        # Unexpected application failure.
        except Exception as error:

            logger.exception(
                "agent_internal_error "
                "request_id=%s",
                request_id,
            )

            raise HTTPException(
                status_code=500,
                detail=(
                    "Internal server error."
                ),
            ) from error

        logger.info(
            "agent_success "
            "request_id=%s tool=%s",
            request_id,
            result["tool_call"]["tool"],
        )

        return ChatResponse(
            **result
        )

    return app


# Production application
app = create_app()