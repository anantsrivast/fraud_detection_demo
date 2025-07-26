import asyncio
from typing import Dict, Any, List
from agents.base_agent import BaseAgent
from services.cloud_mongodb_service import CloudMongoDBService
from services.vector_search_service import VectorSearchService
from services.fraud_signature_service import FraudSignatureService
from models.transaction import Transaction
import logging

logger = logging.getLogger(__name__)

class SimilaritySearchAgent(BaseAgent):
    def __init__(self, mongodb_service: CloudMongoDBService, 
                 vector_search_service: VectorSearchService,
                 signature_service: FraudSignatureService,
                 agent_id: str = None):
        super().__init__(agent_id)
        self.mongodb_service = mongodb_service
        self.vector_search_service = vector_search_service
        self.signature_service = signature_service
    
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Search for similar fraud cases using vector similarity"""
        try:
            transaction_data = data.get("transaction")
            if not transaction_data:
                raise ValueError("No transaction data provided")
            
            transaction = Transaction(**transaction_data)
            
            # Generate fraud signatures
            loop = asyncio.get_event_loop()
            signatures = await loop.run_in_executor(
                None,
                self.signature_service.generate_fraud_signatures,
                transaction
            )
            
            # Encode signatures to vectors
            vectors = await loop.run_in_executor(
                None,
                self.vector_search_service.encode_signatures,
                signatures
            )
            
            # Search for similar cases
            similar_cases = []
            for vector in vectors:
                cases = await loop.run_in_executor(
                    None,
                    self.mongodb_service.search_similar_signatures_atlas,
                    vector,
                    5
                )
                similar_cases.extend(cases)
            
            # Remove duplicates and rank by similarity
            unique_cases = self._deduplicate_cases(similar_cases)
            
            result = {
                "transaction_id": transaction.transaction_id,
                "customer_id": transaction.customer_id,
                "fraud_signatures": signatures,
                "similar_cases": unique_cases,
                "similarity_search_stats": {
                    "signatures_generated": len(signatures),
                    "vectors_created": len(vectors),
                    "similar_cases_found": len(unique_cases),
                    "search_completed_at": self.last_activity.isoformat()
                }
            }
            
            # Add similarity insights
            if unique_cases:
                result["similarity_insights"] = self._analyze_similarity_patterns(unique_cases)
            
            logger.info(
                f"Similarity search completed for {transaction.transaction_id}: "
                f"found {len(unique_cases)} similar cases"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            raise
    
    def _deduplicate_cases(self, cases: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate cases and sort by similarity"""
        unique_cases = {}
        
        for case in cases:
            case_id = case.get('case_id')
            similarity_score = case.get('similarity_score', 0)
            
            if case_id not in unique_cases or similarity_score > unique_cases[case_id]['similarity_score']:
                unique_cases[case_id] = case
        
        # Sort by similarity score descending
        sorted_cases = sorted(unique_cases.values(), 
                            key=lambda x: x.get('similarity_score', 0), 
                            reverse=True)
        
        return sorted_cases[:10]  # Return top 10 most similar cases
    
    def _analyze_similarity_patterns(self, similar_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze patterns in similar cases"""
        if not similar_cases:
            return {}
        
        # Calculate average similarity score
        avg_similarity = sum(case.get('similarity_score', 0) for case in similar_cases) / len(similar_cases)
        
        # Count signature types
        signature_types = {}
        for case in similar_cases:
            sig_type = case.get('signature_type', 'unknown')
            signature_types[sig_type] = signature_types.get(sig_type, 0) + 1
        
        # Find most common customer patterns
        customers = [case.get('customer_id') for case in similar_cases if case.get('customer_id')]
        repeat_customers = len(customers) - len(set(customers))
        
        return {
            "average_similarity_score": avg_similarity,
            "highest_similarity_score": max(case.get('similarity_score', 0) for case in similar_cases),
            "signature_type_distribution": signature_types,
            "repeat_customer_cases": repeat_customers,
            "total_similar_cases": len(similar_cases),
            "high_confidence_matches": len([c for c in similar_cases if c.get('similarity_score', 0) > 0.9])
        }
