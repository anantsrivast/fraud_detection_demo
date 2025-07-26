from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum

class MerchantCategory(str, Enum):
    online = "online"
    atm = "atm"
    retail = "retail"
    travel = "travel"

class Location(BaseModel):
    city: Optional[str]
    country: Optional[str]
    latitude: Optional[float]
    longitude: Optional[float]
    timezone: Optional[str]


class Transaction(BaseModel):
    transaction_id: str
    customer_id: str
    amount: float = Field(default=None, alias="Amount") # Set default to None
    merchant: str
    merchant_category: str
    device_type: str
    ip_address: str
    location: Location
    timestamp: datetime
    is_fraud: int = Field(default=None, alias="Class")  # Set default to None
    time_index: float = Field(default=None, alias="Time")  # Set default to None

    class Config:
        allow_population_by_field_name = True
        populate_by_name = True
        alias_arbitrary = True




class FraudCase(BaseModel):
    case_id: str
    customer_id: str
    transaction_id: str
    fraud_signatures: List[str]
    action_taken: str
    resolution_notes: str
    timestamp: datetime

class CustomerMemory(BaseModel):
    customer_id: str
    last_fraud_report: datetime
    risk_score: float
    recent_cases: List[str]
