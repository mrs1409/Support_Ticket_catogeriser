from pydantic import BaseModel
from typing import Dict, List, Optional

class TicketRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "My account is locked and I cannot access anything urgently!"
            }
        }

class TicketResponse(BaseModel):
    category: str
    confidence: float
    all_scores: Dict[str, float]
    urgency: str
    urgency_reason: str
    model_name: str
    accuracy: Optional[float] = None
    f1_weighted: Optional[float] = None

class ModelInfoResponse(BaseModel):
    best_model_name: str
    accuracy: float
    f1_weighted: float
    f1_macro: float
    n_train: int
    n_test: int
    categories: List[str]
    model_comparison: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: Optional[str] = None
