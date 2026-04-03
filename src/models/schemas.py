from pydantic import BaseModel, field_validator
from typing import List, Literal, Optional


class DocumentRequest(BaseModel):
    fileName: str
    fileType: Literal["pdf", "docx", "image"]
    fileBase64: str

    @field_validator("fileName")
    @classmethod
    def filename_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("fileName must not be empty")
        return v.strip()

    @field_validator("fileBase64")
    @classmethod
    def base64_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("fileBase64 must not be empty")
        return v.strip()


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
