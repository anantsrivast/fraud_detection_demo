
from typing import Dict, Any

class RuleBasedFraudDetector:
    def __init__(self):
        # Define internal parameters or thresholds here
        self.high_risk_countries = {"GN", "SV", "KN", "KH", "RS"}
        self.suspicious_categories = {"electronics", "luxury"}
        self.suspicious_device_type = "mobile"
        self.amount_threshold = 2000

    def predict(self, transaction: Dict[str, Any]) -> bool:
        amount = transaction.get("Amount", 0)
        location = transaction.get("location", {})
        country = location.get("country", "")
        device = transaction.get("device_type", "").lower()
        category = transaction.get("merchant_category", "").lower()

        is_high_amount = amount > self.amount_threshold
        is_foreign_country = country in self.high_risk_countries
        is_device_category_risky = device == self.suspicious_device_type and category in self.suspicious_categories

        return is_high_amount or is_foreign_country or is_device_category_risky
