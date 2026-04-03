import os
import logging
from fastapi import Request, HTTPException, Depends
from fastapi.security import APIKeyHeader

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)


async def verify_api_key(api_key: str = Depends(api_key_header)) -> str:
    """
    Dependency that validates the x-api-key header.
    Raises 401 if missing or invalid.
    """
    expected_key = os.getenv("API_KEY", "")

    if not api_key:
        logger.warning("Request rejected: missing x-api-key header")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Missing API key. Provide x-api-key header.",
        )

    if not expected_key:
        logger.error("Server misconfiguration: API_KEY environment variable not set")
        raise HTTPException(
            status_code=500,
            detail="Server configuration error.",
        )

    if api_key != expected_key:
        logger.warning(f"Request rejected: invalid API key provided")
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid API key.",
        )

    return api_key
