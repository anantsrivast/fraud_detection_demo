from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.collection import Collection
from pymongo.errors import ConnectionFailure, OperationFailure
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from models.transaction import Transaction, CustomerMemory, FraudCase
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class CloudMongoDBService:
    def __init__(self):
        self.client: Optional[MongoClient] = None
        self.db = None
        self.customers: Optional[Collection] = None
        self.cases: Optional[Collection] = None
        self.signatures: Optional[Collection] = None
        
        self._connect()
    
    def _connect(self):
        """Connect to MongoDB Atlas"""
        try:
            if not settings.MONGODB_URI:
                raise ValueError("MongoDB URI not configured")
            
            self.client = MongoClient(
                settings.MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000,
                connectTimeoutMS=20000
            )
            
            # Test connection
            self.client.admin.command('ping')
            
            self.db = self.client[settings.MONGODB_DATABASE]
            self.customers = self.db[settings.MONGODB_COLLECTION_CUSTOMERS]
            self.cases = self.db[settings.MONGODB_COLLECTION_CASES]
            self.signatures = self.db[settings.MONGODB_COLLECTION_SIGNATURES]
            
            logger.info("MongoDB Atlas connection established successfully")
            
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB Atlas: {e}")
            raise
        except Exception as e:
            logger.error(f"MongoDB Atlas connection error: {e}")
            raise
    
    def health_check(self) -> bool:
        """Check if MongoDB Atlas connection is healthy"""
        try:
            self.client.admin.command('ping')
            return True
        except Exception:
            return False
    
    def check_duplicate_complaint(self, customer_id: str, transaction: Transaction) -> bool:
        """Check if customer has similar recent complaint using Atlas search"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=settings.DUPLICATE_DETECTION_HOURS)
            
            # Check customer memory with optimized query
            customer_memory = self.customers.find_one(
                {
                    "customer_id": customer_id,
                    "last_fraud_report": {"$gte": cutoff_time}
                },
                {"recent_cases": 1, "last_fraud_report": 1}
            )
            
            if not customer_memory:
                return False
            
            # Check for similar recent cases using aggregation pipeline
            pipeline = [
                {
                    "$match": {
                        "customer_id": customer_id,
                        "timestamp": {"$gte": cutoff_time}
                    }
                },
                {
                    "$lookup": {
                        "from": "transactions",
                        "localField": "transaction_id",
                        "foreignField": "transaction_id",
                        "as": "transaction_data"
                    }
                },
                {
                    "$match": {
                        "transaction_data.amount": {
                            "$gte": transaction.amount * 0.9,
                            "$lte": transaction.amount * 1.1
                        },
                        "transaction_data.merchant_category": transaction.merchant_category.value
                    }
                }
            ]
            
            similar_cases = list(self.cases.aggregate(pipeline))
            
            if similar_cases:
                logger.info(f"Duplicate complaint detected for customer {customer_id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking duplicate complaint: {e}")
            return False
    
    def update_customer_memory(self, customer_id: str, transaction: Transaction, is_fraud: bool):
        """Update customer memory with atomic operations"""
        try:
            update_operations = {
                "$set": {
                    "customer_id": customer_id,
                    "last_updated": datetime.now()
                },
                "$inc": {
                    "risk_score": 0.1 if is_fraud else -0.01,
                    "transaction_count": 1
                },
                "$max": {
                    "max_transaction_amount": transaction.amount
                }
            }
            
            if is_fraud:
                update_operations["$set"]["last_fraud_report"] = datetime.now()
                update_operations["$inc"]["fraud_count"] = 1
            
            # Atomic upsert operation
            result = self.customers.update_one(
                {"customer_id": customer_id},
                update_operations,
                upsert=True
            )
            
            logger.info(f"Customer memory updated for {customer_id}")
            
        except Exception as e:
            logger.error(f"Error updating customer memory: {e}")
            raise
    
    def store_fraud_case(self, fraud_case: FraudCase):
        """Store fraud case with transaction"""
        try:
            # Use MongoDB transaction for consistency
            with self.client.start_session() as session:
                with session.start_transaction():
                    # Insert fraud case
                    self.cases.insert_one(fraud_case.model_dump(), session=session)
                    
                    # Update customer memory with recent case
                    self.customers.update_one(
                        {"customer_id": fraud_case.customer_id},
                        {
                            "$push": {
                                "recent_cases": {
                                    "$each": [fraud_case.case_id],
                                    "$slice": -10  # Keep only last 10 cases
                                }
                            },
                            "$set": {
                                "last_case_date": fraud_case.timestamp
                            }
                        },
                        upsert=True,
                        session=session
                    )
            
            logger.info(f"Fraud case {fraud_case.case_id} stored successfully")
            
        except Exception as e:
            logger.error(f"Error storing fraud case: {e}")
            raise
    
    def store_fraud_signatures(self, case_id: str, customer_id: str, 
                              signatures: List[str], vectors: List[List[float]]):
        """Store fraud signatures with vectors for Atlas Vector Search"""
        try:
            signature_docs = []
            
            for i, (signature, vector) in enumerate(zip(signatures, vectors)):
                signature_doc = {
                    "case_id": case_id,
                    "customer_id": customer_id,
                    "signature_text": signature,
                    "signature_type": self._get_signature_type(i),
                    "vector_embedding": vector,  # For Atlas Vector Search
                    "timestamp": datetime.now(),
                    "metadata": {
                        "vector_model": settings.EMBEDDING_MODEL,
                        "vector_dimension": len(vector)
                    }
                }
                signature_docs.append(signature_doc)
            
            # Bulk insert for efficiency
            if signature_docs:
                self.signatures.insert_many(signature_docs)
                logger.info(f"Stored {len(signature_docs)} fraud signatures for case {case_id}")
            
        except Exception as e:
            logger.error(f"Error storing fraud signatures: {e}")
            raise
    
    def search_similar_signatures_atlas(self, query_vector: List[float], 
                                       limit: int = 5) -> List[Dict[str, Any]]:
        """Search similar signatures using MongoDB Atlas Vector Search"""
        try:
            # Atlas Vector Search aggregation pipeline
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": settings.VECTOR_INDEX_NAME,
                        "path": "vector_embedding",
                        "queryVector": query_vector,
                        "numCandidates": limit * 2,
                        "limit": limit
                    }
                },
                {
                    "$match": {
                        "timestamp": {
                            "$gte": datetime.now() - timedelta(days=365)  # Last year
                        }
                    }
                },
                {
                    "$addFields": {
                        "similarity_score": {"$meta": "vectorSearchScore"}
                    }
                },
                {
                    "$match": {
                        "similarity_score": {"$gte": settings.SIMILARITY_THRESHOLD}
                    }
                },
                {
                    "$sort": {"similarity_score": -1}
                }
            ]
            
            similar_signatures = list(self.signatures.aggregate(pipeline))
            logger.info(f"Found {len(similar_signatures)} similar signatures")
            
            return similar_signatures
            
        except Exception as e:
            logger.warning(f"Atlas Vector Search failed, falling back to basic search: {e}")
            return self._basic_similarity_search(query_vector, limit)
    
    def _basic_similarity_search(self, query_vector: List[float], 
                                limit: int) -> List[Dict[str, Any]]:
        """Fallback basic similarity search"""
        try:
            # Simple similarity search without vector index
            recent_signatures = self.signatures.find({
                "timestamp": {"$gte": datetime.now() - timedelta(days=30)}
            }).limit(100)
            
            similar_cases = []
            for sig_doc in recent_signatures:
                if 'vector_embedding' in sig_doc:
                    similarity = self._cosine_similarity(query_vector, sig_doc['vector_embedding'])
                    if similarity > settings.SIMILARITY_THRESHOLD:
                        sig_doc['similarity_score'] = similarity
                        similar_cases.append(sig_doc)
            
            # Sort by similarity and return top results
            similar_cases.sort(key=lambda x: x['similarity_score'], reverse=True)
            return similar_cases[:limit]
            
        except Exception as e:
            logger.error(f"Error in basic similarity search: {e}")
            return []
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            vec1_np = np.array(vec1)
            vec2_np = np.array(vec2)
            
            dot_product = np.dot(vec1_np, vec2_np)
            norm1 = np.linalg.norm(vec1_np)
            norm2 = np.linalg.norm(vec2_np)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            return float(dot_product / (norm1 * norm2))
        except Exception:
            return 0.0
    
    def _get_signature_type(self, index: int) -> str:
        """Get signature type based on index"""
        types = ["behavioral", "location", "temporal", "monetary"]
        return types[index] if index < len(types) else "other"
    
    def get_customer_history(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive customer history"""
        try:
            # Use aggregation to get rich customer data
            pipeline = [
                {"$match": {"customer_id": customer_id}},
                {
                    "$lookup": {
                        "from": settings.MONGODB_COLLECTION_CASES,
                        "localField": "customer_id",
                        "foreignField": "customer_id",
                        "as": "fraud_cases",
                        "pipeline": [
                            {"$sort": {"timestamp": -1}},
                            {"$limit": 5}
                        ]
                    }
                }
            ]
            
            result = list(self.customers.aggregate(pipeline))
            return result[0] if result else None
            
        except Exception as e:
            logger.error(f"Error getting customer history: {e}")
            return None
    
    def close(self):
        """Close database connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB Atlas connection closed")
