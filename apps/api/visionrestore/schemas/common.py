from datetime import datetime, timezone
from typing import Any
from uuid import uuid4
from pydantic import BaseModel

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

class ApiResponse(BaseModel):
    success: bool = True
    data: Any = None
    request_id: str
    timestamp: str

def ok(data: Any = None) -> ApiResponse:
    return ApiResponse(success=True, data=data, request_id=str(uuid4()), timestamp=now_iso())

class ApiError(BaseModel):
    success: bool = False
    code: str
    message: str
    details: dict[str, Any] = {}
    request_id: str
    timestamp: str
