import asyncio
from typing import Dict, Any, Tuple
from agents.base_agent import BaseAgent
from models.fraud_classifier import FraudClassifier
from models.transaction import Transaction
import logging

logger = logging.getLogger(__name__)

class FraudClassificationAgent(BaseAgent):
    def __init__(self, fraud_classifier: FraudClassifier, agent_id: str = None):
        super().__init__(agent_id)
        self.fraud_classifier = fraud_classifier
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Classify transaction as fraudulent or legitimate"""
        try:
            transaction_data = data.get("transaction")
            if not transaction_data:
                raise ValueError("No transaction data provided")
            
            transaction = Transaction(**transaction_data)
            
            # Run classification in executor to avoid blocking
            loop = asyncio.get_event_loop()
            is_fraud, probability = await loop.run_in_executor(
                None,
                self.fraud_classifier.predict,
                transaction
            )
            
            # Determine confidence level
            confidence_level = self._get_confidence_level(probability)
            
            result = {
                "is_fraud": is_fraud,
                "fraud_probability": probability,
                "confidence_level": confidence_level,
                "transaction_id": transaction.transaction_id,
                "customer_id": transaction.customer_id,
                "classification_method": "ml_model" if self.fraud_classifier.is_trained else "rule_based",
                "classified_at": self.last_activity.isoformat()
            }
            
            # Add risk indicators
            risk_indicators = self._analyze_risk_factors(transaction, probability)
            result["risk_indicators"] = risk_indicators
            
            logger.info(
                f"Classification completed for {transaction.transaction_id}: "
                f"fraud={is_fraud}, probability={probability:.3f}, confidence={confidence_level}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in fraud classification: {e}")
            raise
    
    def _get_confidence_level(self, probability: float) -> str:
        """Determine confidence level based on probability"""
        if probability >= 0.8 or probability <= 0.2:
            return "HIGH"
        elif probability >= 0.6 or probability <= 0.4:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _analyze_risk_factors(self, transaction: Transaction, probability: float) -> Dict[str, Any]:
        """Analyze specific risk factors"""
        risk_factors = {
            "high_amount": transaction.amount > 2000,
            "unusual_time": self._is_unusual_time(transaction),
            "high_risk_merchant": transaction.merchant_category.value in ["online", "atm"],
            "probability_score": probability
        }
        
        risk_factors["total_risk_factors"] = sum(1 for v in risk_factors.values() if v is True)
        
        return risk_factors
    
    def _is_unusual_time(self, transaction: Transaction) -> bool:
        """Check if transaction time is unusual"""
        hour = transaction.timestamp.hour
        return hour < 6 or hour > 22  # Between 10 PM and 6 AM
