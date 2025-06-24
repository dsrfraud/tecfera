from pydantic import BaseModel
from typing import Optional, Any

class BaseResponse(BaseModel):
    status: str  # "success" or "error"
    message: Optional[str] = None
    data: Optional[Any] = None
