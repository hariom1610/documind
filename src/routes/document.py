import base64
import logging
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from src.models.schemas import DocumentRequest, DocumentResponse, EntitiesModel
from src.services.auth import verify_api_key
from src.services.extractor import extract_text
from src.services.ai_service import analyze_document

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post(
    "/document-analyze",
    response_model=DocumentResponse,
    summary="Analyze a document",
    description=(
        "Accepts a Base64-encoded PDF, DOCX, or image file. "
        "Returns AI-generated summary, named entity extraction, and sentiment analysis."
    ),
    responses={
        200: {"description": "Analysis successful"},
        400: {"description": "Bad request — invalid file, format, or base64"},
        401: {"description": "Unauthorized — invalid or missing API key"},
        500: {"description": "Internal server error during processing"},
    },
)
async def analyze_document_endpoint(
    request: DocumentRequest,
    _: str = Depends(verify_api_key),
) -> JSONResponse:
    """
    Main document analysis endpoint.

    Pipeline:
      1. Decode base64 → raw bytes
      2. Extract text (PDF / DOCX / image via OCR)
      3. Send to Claude for summarization, entity extraction, sentiment
      4. Return structured JSON response
    """
    file_name = request.fileName
    file_type = request.fileType

    # --- Step 1: Decode base64 ---
    try:
        file_bytes = base64.b64decode(request.fileBase64, validate=True)
    except Exception:
        logger.warning(f"Invalid base64 for file: {file_name}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "fileName": file_name,
                "message": "Invalid base64 encoding. Ensure the file is properly encoded.",
            },
        )

    if len(file_bytes) == 0:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "fileName": file_name,
                "message": "Decoded file is empty.",
            },
        )

    # --- Step 2: Extract text ---
    try:
        extracted_text = extract_text(file_bytes, file_type)
    except ValueError as e:
        logger.warning(f"Text extraction failed for {file_name}: {e}")
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "fileName": file_name,
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Unexpected extraction error for {file_name}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "fileName": file_name,
                "message": "An unexpected error occurred during text extraction.",
            },
        )

    # --- Step 3: AI Analysis ---
    try:
        analysis = analyze_document(extracted_text, file_name)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={
                "status": "error",
                "fileName": file_name,
                "message": str(e),
            },
        )
    except RuntimeError as e:
        logger.error(f"AI analysis failed for {file_name}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "fileName": file_name,
                "message": str(e),
            },
        )
    except Exception as e:
        logger.error(f"Unexpected AI error for {file_name}: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "status": "error",
                "fileName": file_name,
                "message": "An unexpected error occurred during AI analysis.",
            },
        )

    # --- Step 4: Build and return response ---
    entities_data = analysis.get("entities", {})
    response = DocumentResponse(
        status="success",
        fileName=file_name,
        summary=analysis.get("summary", ""),
        entities=EntitiesModel(
            names=entities_data.get("names", []),
            dates=entities_data.get("dates", []),
            organizations=entities_data.get("organizations", []),
            amounts=entities_data.get("amounts", []),
            locations=entities_data.get("locations", []),
        ),
        sentiment=analysis.get("sentiment", "Neutral"),
    )

    logger.info(f"Successfully processed: {file_name}")
    return JSONResponse(status_code=200, content=response.model_dump(exclude_none=False))
