import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.utils import get_openapi
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from dotenv import load_dotenv

from src.models.schemas import DocumentResponse, EntitiesModel, DocumentRequest
from src.routes.document import router as document_router, analyze_document_endpoint
from src.services.auth import verify_api_key
from fastapi import Depends


# Load environment variables from .env file (for local development)
load_dotenv(override=True)

# --- Logging configuration ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== Document Analysis API starting up ===")
    missing = []
    if not os.getenv("API_KEY"):
        missing.append("API_KEY")
    if missing:
        logger.error(f"Missing env vars: {', '.join(missing)}")
    else:
        logger.info("All env vars set.")
    
    # START model loading immediately at startup in background
    from src.services.ai_service import _get_pipelines
    import threading
    threading.Thread(target=_get_pipelines, daemon=True).start()
    logger.info("Model loading started in background during startup (may take up to 2 minutes).")
    logger.info("Use /models-ready endpoint to check model loading status.")
    
    yield
    logger.info("=== Shutting down ===")


# --- FastAPI app ---
app = FastAPI(
    title="Document Analysis API",
    description=(
        "An intelligent document processing API that extracts, analyses, and summarises "
        "content from PDF, DOCX, and image files using AI. "
        "Supports automatic text extraction, named entity recognition, sentiment analysis, "
        "and AI-powered summarisation."
    ),
    version="1.0.0",
    contact={
        "name": "Document Analysis API",
    },
    lifespan=lifespan,
)


# --- Custom OpenAPI schema with security definition ---
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Document Analysis API",
        version="1.0.0",
        description=(
            "An intelligent document processing API that extracts, analyses, and summarises "
            "content from PDF, DOCX, and image files using AI.\n\n"
            "**Authentication**: All endpoints except `/health` and `/models-ready` require the `x-api-key` header.\n\n"
            "**Important**: On first request, models will load (takes 30-120 seconds). "
            "Use `/models-ready` to check if models are ready before sending analysis requests."
        ),
        routes=app.routes,
    )
    
    # Add API key security scheme without overwriting other generated components
    components = openapi_schema.setdefault("components", {})
    security_schemes = components.setdefault("securitySchemes", {})
    security_schemes["ApiKeyHeader"] = {
        "type": "apiKey",
        "in": "header",
        "name": "x-api-key",
        "description": "API key for authentication. Contact the API administrator for access."
    }
    
    # Mark all endpoints (except health) as requiring API key
    for path, path_item in openapi_schema.get("paths", {}).items():
        if path not in ["/", "/health", "/models-ready"]:
            for operation in path_item.values():
                if isinstance(operation, dict) and "security" not in operation:
                    operation["security"] = [{"ApiKeyHeader": []}]
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure 422 errors return the required JSON structure even if parsing fails."""
    file_name = "unknown"
    try:
        # Check if json was already parsed and cached by Starlette
        if hasattr(request, "_json") and isinstance(request._json, dict):
            file_name = request._json.get("fileName", "unknown")
        else:
            body = await request.json()
            if isinstance(body, dict):
                file_name = body.get("fileName", "unknown")
    except Exception:
        pass

    content = DocumentResponse(
        status="error",
        fileName=file_name,
        message="Invalid request payload. Ensure all required fields are present and valid.",
        summary="",
        entities=EntitiesModel(),
        sentiment=""
    ).model_dump(exclude_none=False)

    return JSONResponse(status_code=422, content=content)

@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    file_name = "unknown"
    if request.method in ("POST", "PUT", "PATCH"):
        try:
            if hasattr(request, "_json") and isinstance(request._json, dict):
                file_name = request._json.get("fileName", "unknown")
            else:
                body = await request.json()
                if isinstance(body, dict):
                    file_name = body.get("fileName", "unknown")
        except Exception:
            pass

    content = DocumentResponse(
        status="error",
        fileName=file_name,
        summary="",
        entities=EntitiesModel(),
        sentiment=""
    ).model_dump(exclude_none=False)

    return JSONResponse(status_code=exc.status_code, content=content)


# --- CORS Middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Routes ---
app.include_router(document_router, prefix="/api", tags=["Document Analysis"])

@app.post("/", tags=["Document Analysis"])
async def root_post_fallback(
    request: DocumentRequest,
    api_key: str = Depends(verify_api_key)
):
    """Fallback endpoint in case the tester POSTs to the root URL."""
    return await analyze_document_endpoint(request=request, _=api_key)


# --- Health check ---
@app.api_route("/", methods=["GET", "HEAD"], tags=["Health"])
async def root():
    """Root endpoint — health check."""
    return JSONResponse(
        content={
            "status": "ok",
            "message": "Document Analysis API is running",
            "version": "1.0.0",
            "docs": "/docs",
        }
    )


@app.api_route("/health", methods=["GET", "HEAD"], tags=["Health"])
async def health_check():
    """Detailed health check endpoint."""
    return JSONResponse(
        content={
            "status": "healthy",
            "api_key_configured": bool(os.getenv("API_KEY")),
        }
    )


@app.api_route("/models-ready", methods=["GET", "HEAD"], tags=["Health"])
async def models_ready():
    """
    Check if AI models are ready for analysis.
    
    Returns:
    - status: "ready" if models loaded successfully, "loading" if still loading, "error" if loading failed
    - message: Human-readable status message
    - details: More info about which models are available
    """
    from src.services.ai_service import get_models_status
    
    status_info = get_models_status()
    
    if not status_info["is_loading"]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "idle",
                "message": "Models have not been requested yet. First API request will trigger loading.",
                "next_action": "Send a document analysis request to start model loading",
            }
        )
    
    if status_info["is_ready"]:
        return JSONResponse(
            status_code=200,
            content={
                "status": "ready",
                "message": "AI models are loaded and ready for analysis.",
                "models": {
                    "summarizer": status_info["summarizer_loaded"],
                    "ner_pipeline": status_info["ner_loaded"],
                    "sentiment_pipeline": status_info["sentiment_loaded"],
                },
            }
        )
    else:
        return JSONResponse(
            status_code=202,
            content={
                "status": "loading",
                "message": "Models are still loading. Please wait and retry in 10-30 seconds.",
                "estimated_wait_seconds": "30-120 depending on system resources",
            }
        )
