from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from services.cloud_mongodb_service import CloudMongoDBService
from config.settings import settings
import logging

logger = logging.getLogger(__name__)

class VectorSearchService:
    def __init__(self, mongodb_service: CloudMongoDBService):
        self.mongodb_service = mongodb_service
        self.encoder = SentenceTransformer(settings.EMBEDDING_MODEL)

    def encode_signatures(self, signatures: List[str]) -> List[List[float]]:
        """Encode fraud signatures into vectors"""
        try:
            vectors = self.encoder.encode(signatures)
            return vectors.tolist()
        except Exception as e:
            logger.error(f"Error encoding signatures: {e}")
            return [[0.0] * settings.VECTOR_DIMENSION] * len(signatures)

    def search_similar_cases(self, query_signatures: List[str], limit: int = 5) -> List[Dict[str, Any]]:
        """Search similar fraud cases using MongoDB Atlas Vector Search"""
        try:
            vectors = self.encode_signatures(query_signatures)
            results = []
            for vector in vectors:
                pipeline = [
                    {
                        "$vectorSearch": {
                            "index": settings.VECTOR_SEARCH_INDEX,
                            "path": "vector",
                            "queryVector": vector,
                            "numCandidates": settings.VECTOR_NUM_CANDIDATES,
                            "limit": limit
                        }
                    },
                    {
                        "$project": {
                            "case_id": 1,
                            "signature_text": 1,
                            "signature_type": 1,
                            "similarity": {"$meta": "vectorSearchScore"},
                            "customer_id": 1,
                            "timestamp": 1
                        }
                    }
                ]
                cursor = self.mongodb_service.signatures.aggregate(pipeline)
                results.extend(cursor)
            # Deduplicate and return top-N
            seen = {}
            for r in results:
                case_id = r['case_id']
                if case_id not in seen or r['similarity'] > seen[case_id]['similarity']:
                    seen[case_id] = r
            return sorted(seen.values(), key=lambda x: x['similarity'], reverse=True)[:limit]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
