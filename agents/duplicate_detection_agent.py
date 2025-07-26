import asyncio
from typing import Dict, Any
from agents.base_agent import BaseAgent
from services.cloud_mongodb_service import CloudMongoDBService
from models.transaction import Transaction
import logging

logger = logging.getLogger(__name__)

class DuplicateDetectionAgent(BaseAgent):
    def __init__(self, mongodb_service: CloudMongoDBService, agent_id: str = None):
        super().__init__(agent_id)
        self.mongodb_service = mongodb_service
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check for duplicate complaints"""
        try:
            transaction_data = data.get("transaction")
            if not transaction_data:
                raise ValueError("No transaction data provided")
            
            transaction = Transaction(**transaction_data)
            
            # Check for duplicate in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            is_duplicate = await loop.run_in_executor(
                None, 
                self.mongodb_service.check_duplicate_complaint,
                transaction.customer_id,
                transaction
            )
            
            result = {
                "is_duplicate": is_duplicate,
                "customer_id": transaction.customer_id,
                "transaction_id": transaction.transaction_id,
                "checked_at": self.last_activity.isoformat()
            }
            
            if is_duplicate:
                result["recommendation"] = "SKIP_PROCESSING"
                result["reason"] = "Duplicate complaint detected within 24 hours"
            else:
                result["recommendation"] = "CONTINUE_PROCESSING"
            
            logger.info(f"Duplicate check completed for {transaction.transaction_id}: {is_duplicate}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in duplicate detection: {e}")
            raise
