import base64
import logging
from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi.responses import JSONResponse

from src.models.schemas import DocumentRequest, DocumentResponse, EntitiesModel
from src.services.auth import verify_api_key
from src.services.extractor import extract_text
from src.services.ai_service import analyze_document

import asyncio
from functools import partial


logger = logging.getLogger(__name__)

router = APIRouter()

def make_error_response(file_name: str, message: str, status_code: int = 400) -> JSONResponse:
    content = DocumentResponse(
        status="error",
        fileName=file_name if file_name else "unknown",
        message=message,
        summary="",
        entities=EntitiesModel(),
        sentiment=""
    ).model_dump(exclude_none=False)
    return JSONResponse(status_code=status_code, content=content)


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
    request: DocumentRequest = Body(
        ..., 
        example={
            "fileName": "contract.pdf",
            "fileType": "pdf",
            "fileBase64": "JVBERi0xLjQKJcfs..."
        }
    ),
    _: str = Depends(verify_api_key),
) -> JSONResponse:
    """
    Main document analysis endpoint.

    Pipeline:
      1. Decode base64 → raw bytes
      2. Extract text (PDF / DOCX / image via OCR)
      3. Send to Claude for summarization, entity extraction, sentiment
      4. Return structured JSON response
      
    Note: First request may take 30-120 seconds while AI models are loading.
    Use /models-ready endpoint to check if models are ready.
    """
    import time
    start_time = time.time()
    file_name = request.fileName
    file_type = request.fileType
    logger.info(f"Processing started for: {file_name} ({file_type})")

    # # --- Step 1: Decode base64 ---
    try:
        b64_string = request.fileBase64.strip()
        
        # Handle default Swagger UI payload
        if b64_string == "string":
            return make_error_response(file_name, "Please provide a valid base64 encoded file instead of the default 'string' value from Swagger UI.", 400)

        # Strip Data URI prefix if present (e.g., 'data:application/pdf;base64,')
        if "," in b64_string:
            b64_string = b64_string.split(",", 1)[1]

        # Remove all whitespace (newlines, tabs, spaces) that might have been added
        b64_string = "".join(b64_string.split())

        # Perform C-level decoded validation, which runs extremely fast and safely
        file_bytes = base64.b64decode(b64_string, validate=True)
        
        MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
        if len(file_bytes) > MAX_FILE_SIZE:
            return make_error_response(file_name, "File too large. Max 10MB.", 413)
    except Exception as e:
        logger.warning(f"Invalid base64 for file {file_name}: {e}")
        return make_error_response(file_name, "Invalid base64 encoding. Ensure the file is properly encoded and does not contain invalid characters. The base64 string may contain newlines or invalid characters.", 400)


    if len(file_bytes) == 0:
        return make_error_response(file_name, "Decoded file is empty.", 400)

    # --- Step 2: Extract text ---
    try:
        loop = asyncio.get_event_loop()
        extracted_text = await loop.run_in_executor(
            None,
            partial(extract_text, file_bytes, file_type)
        )
    except ValueError as e:
        elapsed_time = time.time() - start_time
        logger.warning(f"Text extraction failed for {file_name} after {elapsed_time:.2f}s: {e}")
        return make_error_response(file_name, str(e), 400)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Unexpected extraction error for {file_name} after {elapsed_time:.2f}s: {e}")
        return make_error_response(file_name, "An unexpected error occurred during text extraction.", 500)

    # --- Step 3: AI Analysis ---
    try:
        loop = asyncio.get_event_loop()
        analysis = await loop.run_in_executor(
            None,
            partial(analyze_document, extracted_text, file_name)
        )
    except ValueError as e:
        elapsed_time = time.time() - start_time
        logger.warning(f"Validation error for {file_name} after {elapsed_time:.2f}s: {e}")
        return make_error_response(file_name, str(e), 400)
    except RuntimeError as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        if "120 seconds" in error_msg:
            logger.error(f"Model loading timeout for {file_name} after {elapsed_time:.2f}s")
            return make_error_response(
                file_name, 
                "Models are still loading (timeout). This can happen on first request. Try again in 2-3 minutes. "
                "Check /models-ready endpoint to verify model status.", 
                503
            )
        logger.error(f"AI analysis failed for {file_name} after {elapsed_time:.2f}s: {e}")
        return make_error_response(file_name, str(e), 500)
    except Exception as e:
        elapsed_time = time.time() - start_time
        logger.error(f"Unexpected AI error for {file_name} after {elapsed_time:.2f}s: {e}")
        return make_error_response(file_name, "An unexpected error occurred during AI analysis.", 500)

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

    elapsed_time = time.time() - start_time
    logger.info(f"Successfully processed: {file_name} in {elapsed_time:.2f}s")
    return JSONResponse(status_code=200, content=response.model_dump(exclude_none=False))
