import os
import logging
import hmac 
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(request: Request, api_key: str = Depends(api_key_header)) -> str:
    """
    Dependency that validates the x-api-key header.
    Raises 401 if missing or invalid.
    
    The x-api-key header must be provided in all requests to protected endpoints.
    """
    expected_key = os.getenv("API_KEY", "")

    if not api_key:
        logger.warning(f"Request rejected: missing x-api-key header | Path: {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing API key. Provide x-api-key header.",
            headers={"WWW-Authenticate": "ApiKeyHeader"},
        )

    if not expected_key:
        logger.error("Server misconfiguration: API_KEY environment variable not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error.",
        )

    if not hmac.compare_digest(api_key, expected_key):
        logger.warning(f"Request rejected: invalid API key provided | Path: {request.url.path}")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API key.",
            headers={"WWW-Authenticate": "ApiKeyHeader"},
        )

    logger.debug(f"API key validated successfully for {request.url.path}")
    return api_key
