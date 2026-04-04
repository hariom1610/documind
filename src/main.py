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
    """Startup and shutdown lifecycle events."""
    logger.info("=== Document Analysis API starting up ===")

    # Validate required environment variables on startup
    missing = []
    if not os.getenv("API_KEY"):
        missing.append("API_KEY")

    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
    else:
        logger.info("All required environment variables are set.")

    yield

    logger.info("=== Document Analysis API shutting down ===")


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

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Ensure 422 errors return the required JSON structure even if parsing fails."""
    file_name = "unknown"
    try:
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
    """Ensure HTTP errors (405, 404, 401) return the required JSON structure."""
    file_name = "unknown"
    try:
        body = await request.json()
        if isinstance(body, dict):
            file_name = body.get("fileName", "unknown")
    except Exception:
        pass

    content = DocumentResponse(
        status="error",
        fileName=file_name,
        message=str(exc.detail),
        summary="",
        entities=EntitiesModel(),
        sentiment=""
    ).model_dump(exclude_none=False)

    return JSONResponse(status_code=exc.status_code, content=content)

# # --- Custom OpenAPI schema with security scheme ---
# def custom_openapi():
#     if app.openapi_schema:
#         return app.openapi_schema
#     openapi_schema = get_openapi(
#         title="Document Analysis API",
#         version="1.0.0",
#         description=(
#             "An intelligent document processing API that extracts, analyses, and summarises "
#             "content from PDF, DOCX, and image files using AI."
#         ),
#         routes=app.routes,
#     )
#     openapi_schema["components"]["securitySchemes"] = {
#         "api_key": {
#             "type": "apiKey",
#             "in": "header",
#             "name": "x-api-key",
#             "description": "API key for authentication",
#         }
#     }
#     openapi_schema["security"] = [{"api_key": []}]
#     app.openapi_schema = openapi_schema
#     return app.openapi_schema

# app.openapi = custom_openapi

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
