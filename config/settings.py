import os
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

class Settings:
    # Cloud Kafka Configuration (Confluent Cloud)
    KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS")
    KAFKA_API_KEY = os.getenv("KAFKA_API_KEY")
    KAFKA_API_SECRET = os.getenv("KAFKA_API_SECRET")
    KAFKA_TOPIC = os.getenv("KAFKA_TOPIC", "fraudulent-transactions")
    KAFKA_OUTPUT_TOPIC = os.getenv("KAFKA_OUTPUT_TOPIC", "fraud-results")
    KAFKA_CONSUMER_GROUP = os.getenv("KAFKA_CONSUMER_GROUP", "fraud-detection-agents")

    # Schema Registry Configuration
    SCHEMA_REGISTRY_URL = os.getenv("SCHEMA_REGISTRY_URL")
    SCHEMA_REGISTRY_API_KEY = os.getenv("SCHEMA_REGISTRY_API_KEY")
    SCHEMA_REGISTRY_API_SECRET = os.getenv("SCHEMA_REGISTRY_API_SECRET")
    SCHEMA_REGISTRY_SUBJECT = os.getenv("SCHEMA_REGISTRY_SUBJECT", "fraudulent-transactions-value")
    
    # MongoDB Atlas Configuration
    MONGODB_URI = os.getenv("MONGODB_URI")
    MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "fraud_detection")
    MONGODB_COLLECTION_CUSTOMERS = os.getenv("MONGODB_COLLECTION_CUSTOMERS", "customer_memory")
    MONGODB_COLLECTION_CASES = os.getenv("MONGODB_COLLECTION_CASES", "fraud_cases")
    MONGODB_COLLECTION_SIGNATURES = os.getenv("MONGODB_COLLECTION_SIGNATURES", "fraud_signatures")
    
    # Vector Search Configuration
    VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "fraud_signature_vector_index")
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    VECTOR_DIMENSION = int(os.getenv("VECTOR_DIMENSION", "384"))
    
    # Fraud Detection Configuration
    SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.8"))
    DUPLICATE_DETECTION_HOURS = int(os.getenv("DUPLICATE_DETECTION_HOURS", "24"))
    
    # Model Configuration
    #FRAUD_MODEL_PATH = os.getenv("FRAUD_MODEL_PATH", "models/fraud_classifier.joblib")
    
    # Agent Configuration
    MAX_CONCURRENT_AGENTS = int(os.getenv("MAX_CONCURRENT_AGENTS", "5"))
    AGENT_TIMEOUT_SECONDS = int(os.getenv("AGENT_TIMEOUT_SECONDS", "30"))
    
    # Logging
    #LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
settings = Settings()
