from typing import List, Dict, Any
from models.transaction import Transaction
from datetime import datetime


class FraudSignatureService:
    def __init__(self):
        pass

    def generate_fraud_signatures(self, transaction: Transaction) -> List[str]:
        """Generate multiple descriptive fraud signatures for a transaction"""
        signatures = []

        # Behavioral signature
        behavioral = self._generate_behavioral_signature(transaction)
        signatures.append(behavioral)

        # Location-based signature
        location_sig = self._generate_location_signature(transaction)
        signatures.append(location_sig)

        # Temporal signature
        temporal = self._generate_temporal_signature(transaction)
        signatures.append(temporal)

        # Monetary signature
        monetary = self._generate_monetary_signature(transaction)
        signatures.append(monetary)

        return signatures

    def _generate_behavioral_signature(self, transaction: Transaction) -> str:
        """Generate behavioral fraud signature"""
        return (
            f"Customer behavioral pattern: {transaction.device_type} device usage "
            f"for {transaction.merchant_category} transaction with risk indicators "
            f"derived from transaction behavior"
        )

    def _generate_location_signature(self, transaction: Transaction) -> str:
        """Generate location-based fraud signature"""
        location = transaction.location
        return (
            f"Geographic transaction pattern: {location.city or 'Unknown'}, "
            f"{location.country or 'Unknown'} location with IP {transaction.ip_address} "
            f"showing potential geo-location mismatch or unusual location activity"
        )

    def _generate_temporal_signature(self, transaction: Transaction) -> str:
        """Generate temporal fraud signature"""
        hour = transaction.timestamp.hour
        day = transaction.timestamp.strftime('%A')
        time_category = self._categorize_time(hour)
        return (
            f"Temporal fraud pattern: {day} {time_category} transaction "
            f"at {hour}:00 showing unusual timing for customer transaction behavior"
        )

    def _generate_monetary_signature(self, transaction: Transaction) -> str:
        """Generate monetary fraud signature"""
        amount_category = self._categorize_amount(transaction.amount)
        return (
            f"Monetary fraud pattern: {amount_category} amount transaction "
            f"of ${transaction.amount:.2f} at {transaction.merchant_category} "
            f"merchant showing unusual spending pattern for customer profile"
        )

    def _categorize_time(self, hour: int) -> str:
        """Categorize time of day"""
        if 6 <= hour < 12:
            return "morning"
        elif 12 <= hour < 18:
            return "afternoon"
        elif 18 <= hour < 22:
            return "evening"
        else:
            return "night"

    def _categorize_amount(self, amount: float) -> str:
        """Categorize transaction amount"""
        if amount < 50:
            return "small"
        elif amount < 500:
            return "medium"
        elif amount < 2000:
            return "large"
        else:
            return "very large"

