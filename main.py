import asyncio
import logging
import sys
from datetime import datetime
from config.settings import settings
from setup.cloud_setup import CloudSetup
from models.fraud_classifier import FraudClassifier
from data_generation.synthetic_data_generator import SyntheticDataGenerator
from services.cloud_kafka_service import CloudKafkaService
from services.cloud_mongodb_service import CloudMongoDBService
from services.vector_search_service import VectorSearchService
from services.fraud_signature_service import FraudSignatureService
from agents.duplicate_detection_agent import DuplicateDetectionAgent
from agents.fraud_classification_agent import FraudClassificationAgent
from agents.similarity_search_agent import SimilaritySearchAgent
from agents.action_recommendation_agent import ActionRecommendationAgent
from agents.data_storage_agent import DataStorageAgent
from workflows.multi_agent_workflow import MultiAgentFraudWorkflow
from models.transaction import Transaction

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('fraud_detection.log')
    ]
)
logger = logging.getLogger(__name__)

async def main():
    logger.info("Starting Cloud Fraud Detection System")
    settings.validate()

    # Cloud setup
    cloud_setup = CloudSetup()
    cloud_setup.setup_mongodb_atlas()
    cloud_setup.setup_confluent_kafka()

    # Core services
    mongodb_service = CloudMongoDBService()
    kafka_service = CloudKafkaService()
    vector_search_service = VectorSearchService(mongodb_service)
    signature_service = FraudSignatureService()
    fraud_classifier = FraudClassifier(settings.FRAUD_MODEL_PATH)

    # Agents
    duplicate_agent = DuplicateDetectionAgent(mongodb_service)
    classification_agent = FraudClassificationAgent(fraud_classifier)
    similarity_agent = SimilaritySearchAgent(mongodb_service, vector_search_service, signature_service)
    recommendation_agent = ActionRecommendationAgent()
    storage_agent = DataStorageAgent(mongodb_service)

    # Workflow
    workflow = MultiAgentFraudWorkflow(
        duplicate_agent,
        classification_agent,
        similarity_agent,
        recommendation_agent,
        storage_agent
    )

    # Generate a synthetic transaction for demo
    generator = SyntheticDataGenerator()
    transaction = generator.generate_transaction("cust001")

    logger.info(f"Processing transaction {transaction.transaction_id}")
    result = await workflow.process_transaction(transaction)
    logger.info(f"Workflow result: {result['workflow_status']}")

if __name__ == "__main__":
    asyncio.run(main())
