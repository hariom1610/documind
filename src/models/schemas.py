from pydantic import BaseModel, Field, field_validator
from typing import List, Literal, Optional


class DocumentRequest(BaseModel):
    fileName: str = Field(...)
    fileType: Literal["pdf", "docx", "image"] = Field(...)
    fileBase64: str = Field(
        ..., 
        description="Base64-encoded contents of the uploaded file. Replace this with your actual PDF/DOCX/image payload."
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "fileName": "contract.pdf",
                    "fileType": "pdf",
                    "fileBase64": "JVBERi0xLjQKJcfs..."
                }
            ]
        }
    }

    @field_validator("fileName")
    @classmethod
    def filename_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("fileName must not be empty")
        return v.strip()

    @field_validator("fileBase64")
    @classmethod
    def base64_must_be_valid_size(cls, v: str) -> str:
        v_stripped = v.strip()
        if not v_stripped:
            raise ValueError("fileBase64 must not be empty")
        # 10MB binary limit ~13.3MB base64. Reject anything over 15MB base64 characters.
        if len(v_stripped) > 15 * 1024 * 1024:
            raise ValueError("File payload too large. Base64 content must be under 15MB (approx 10MB decoded).")
        return v_stripped



class EntitiesModel(BaseModel):
    names: List[str] = []
    dates: List[str] = []
    organizations: List[str] = []
    amounts: List[str] = []
    locations: List[str] = []


class DocumentResponse(BaseModel):
    status: str
    fileName: str
    summary: Optional[str] = None
    entities: Optional[EntitiesModel] = None
    sentiment: Optional[str] = None
    message: Optional[str] = None
