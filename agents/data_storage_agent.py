import asyncio
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.cloud_mongodb_service import CloudMongoDBService
from models.transaction import Transaction, FraudCase
import logging
import uuid
from datetime import datetime

logger = logging.getLogger(__name__)

class DataStorageAgent(BaseAgent):
    def __init__(self, mongodb_service: CloudMongoDBService, agent_id: str = None):
        super().__init__(agent_id)
        self.mongodb_service = mongodb_service
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Store fraud analysis results and update customer data"""
        try:
            transaction_data = data.get("transaction")
            fraud_analysis = data.get("fraud_analysis", {})
            similarity_data = data.get("similarity_data", {})
            recommendation = data.get("recommendation", {})
            
            if not transaction_data:
                raise ValueError("No transaction data provided")
            
            transaction = Transaction(**transaction_data)
            
            # Update customer memory
            await self._update_customer_memory(transaction, fraud_analysis)
            
            # Store fraud case if fraud detected
            case_id = None
            if fraud_analysis.get("is_fraud", False):
                case_id = await self._store_fraud_case(
                    transaction, fraud_analysis, similarity_data, recommendation
                )
            
            # Store fraud signatures if available
            signatures_stored = 0
            if similarity_data.get("fraud_signatures"):
                signatures_stored = await self._store_fraud_signatures(
                    case_id or str(uuid.uuid4()),
                    transaction,
                    similarity_data
                )
            
            result = {
                "transaction_id": transaction.transaction_id,
                "customer_id": transaction.customer_id,
                "storage_results": {
                    "customer_memory_updated": True,
                    "fraud_case_stored": case_id is not None,
                    "fraud_case_id": case_id,
                    "signatures_stored": signatures_stored,
                    "storage_completed_at": self.last_activity.isoformat()
                }
            }
            
            logger.info(
                f"Data storage completed for {transaction.transaction_id}: "
                f"case_stored={case_id is not None}, signatures={signatures_stored}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in data storage: {e}")
            raise
    
    async def _update_customer_memory(self, transaction: Transaction, 
                                    fraud_analysis: Dict[str, Any]):
        """Update customer memory in database"""
        try:
            is_fraud = fraud_analysis.get("is_fraud", False)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.mongodb_service.update_customer_memory,
                transaction.customer_id,
                transaction,
                is_fraud
            )
            
        except Exception as e:
            logger.error(f"Error updating customer memory: {e}")
            raise
    
    async def _store_fraud_case(self, transaction: Transaction,
                              fraud_analysis: Dict[str, Any],
                              similarity_data: Dict[str, Any],
                              recommendation: Dict[str, Any]) -> str:
        """Store fraud case in database"""
        try:
            case_id = str(uuid.uuid4())
            
            fraud_case = FraudCase(
                case_id=case_id,
                customer_id=transaction.customer_id,
                transaction_id=transaction.transaction_id,
                fraud_signatures=similarity_data.get("fraud_signatures", []),
                action_taken=recommendation.get("recommended_action", "Unknown"),
                resolution_notes=self._create_resolution_notes(
                    fraud_analysis, similarity_data, recommendation
                ),
                timestamp=datetime.now()
            )
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self.mongodb_service.store_fraud_case,
                fraud_case
            )
            
            return case_id
            
        except Exception as e:
            logger.error(f"Error storing fraud case: {e}")
            raise
    
    async def _store_fraud_signatures(self, case_id: str, transaction: Transaction,
                                    similarity_data: Dict[str, Any]) -> int:
        """Store fraud signatures with vectors"""
        try:
            signatures = similarity_data.get("fraud_signatures", [])
            if not signatures:
                return 0
            
            # Get vectors from similarity service (should be in similarity_data)
            from services.vector_search_service import VectorSearchService
            from services.cloud_mongodb_service import CloudMongoDBService
            
            # Create temporary vector service for encoding
            vector_service = VectorSearchService(self.mongodb_service)
            
            loop = asyncio.get_event_loop()
            vectors = await loop.run_in_executor(
                None,
                vector_service.encode_signatures,
                signatures
            )
            
            # Store signatures with vectors
            await loop.run_in_executor(
                None,
                self.mongodb_service.store_fraud_signatures,
                case_id,
                transaction.customer_id,
                signatures,
                vectors
            )
            
            return len(signatures)
            
        except Exception as e:
            logger.error(f"Error storing fraud signatures: {e}")
            return 0
    
    def _create_resolution_notes(self, fraud_analysis: Dict[str, Any],
                               similarity_data: Dict[str, Any],
                               recommendation: Dict[str, Any]) -> str:
        """Create comprehensive resolution notes"""
        
        notes = []
        
        # Fraud analysis summary
        fraud_prob = fraud_analysis.get("fraud_probability", 0.0)
        confidence = fraud_analysis.get("confidence_level", "UNKNOWN")
        notes.append(f"Fraud probability: {fraud_prob:.2%} (confidence: {confidence})")
        
        # Risk indicators
        risk_indicators = fraud_analysis.get("risk_indicators", {})
        if risk_indicators.get("total_risk_factors", 0) > 0:
            notes.append(f"Risk factors detected: {risk_indicators['total_risk_factors']}")
        
        # Similar cases
        similar_cases = similarity_data.get("similar_cases", [])
        if similar_cases:
            high_sim = len([c for c in similar_cases if c.get('similarity_score', 0) > 0.9])
            notes.append(f"Similar historical cases: {len(similar_cases)} (high similarity: {high_sim})")
        
        # Recommendation details
        action = recommendation.get("recommended_action", "Unknown")
        rec_confidence = recommendation.get("confidence_score", 0.0)
        notes.append(f"Recommended action: {action} (confidence: {rec_confidence:.2%})")
        
        # Additional context
        notes.append(f"Analysis completed at {datetime.now().isoformat()}")
        
        return " | ".join(notes)
