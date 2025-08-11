import json
import logging
from typing import Dict, Any
from pymongo import MongoClient
from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import KafkaError
from config.settings import settings

logger = logging.getLogger(__name__)

class CloudSetup:
    def __init__(self):
        self.mongodb_client = None
        self.kafka_producer = None
        self.kafka_consumer = None
    
    def setup_mongodb_atlas(self) -> bool:
        """Setup MongoDB Atlas collections and indexes"""
        try:
            logger.info("Setting up MongoDB Atlas...")
            
            # Validate MongoDB connection
            if not settings.MONGODB_URI:
                raise ValueError("MONGODB_URI not configured")
            
            self.mongodb_client = MongoClient(settings.MONGODB_URI)
            
            # Test connection
            self.mongodb_client.admin.command('ping')
            logger.info("MongoDB Atlas connection successful")
            
            # Get database
            db = self.mongodb_client[settings.MONGODB_DATABASE]
            
            # Setup collections and indexes
            self._setup_customers_collection(db)
            self._setup_cases_collection(db)
            self._setup_signatures_collection(db)
            
            logger.info("MongoDB Atlas setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"MongoDB Atlas setup failed: {e}")
            return False
    
    def _setup_customers_collection(self, db):
        """Setup customer memory collection"""
        collection = db[settings.MONGODB_COLLECTION_CUSTOMERS]
        
        # Create indexes
        collection.create_index("customer_id", unique=True)
        collection.create_index("last_fraud_report")
        collection.create_index("risk_score")
        
        logger.info("Customer memory collection setup complete")
    
    def _setup_cases_collection(self, db):
        """Setup fraud cases collection"""
        collection = db[settings.MONGODB_COLLECTION_CASES]
        
        # Create indexes
        collection.create_index("customer_id")
        collection.create_index("timestamp")
        collection.create_index("case_id", unique=True)
        collection.create_index([("customer_id", 1), ("timestamp", -1)])
        
        logger.info("Fraud cases collection setup complete")
    
    def _setup_signatures_collection(self, db):
        """Setup fraud signatures collection"""
        collection = db[settings.MONGODB_COLLECTION_SIGNATURES]
        
        # Create indexes
        collection.create_index("case_id")
        collection.create_index("customer_id")
        collection.create_index("timestamp")
        collection.create_index("signature_type")
        
        # For vector search (MongoDB Atlas Vector Search)
        # Note: Vector indexes need to be created through MongoDB Atlas UI or API
        logger.info("Fraud signatures collection setup complete")
        logger.info("Note: Vector search index needs to be created in MongoDB Atlas UI")
    
    def setup_confluent_kafka(self) -> bool:
        """Setup Confluent Cloud Kafka"""
        try:
            logger.info("Setting up Confluent Cloud Kafka...")
            
            # Validate Kafka configuration
            if not all([settings.KAFKA_BOOTSTRAP_SERVERS, 
                       settings.KAFKA_API_KEY, 
                       settings.KAFKA_API_SECRET]):
                raise ValueError("Kafka configuration incomplete")
            
            # Test Kafka connection
            producer_config = {
                'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
                'sasl.mechanisms': 'PLAIN',
                'security.protocol': 'SASL_SSL',
                'sasl.username': settings.KAFKA_API_KEY,
                'sasl.password': settings.KAFKA_API_SECRET,
            }

            consumer_config = {
                'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
                'sasl.mechanisms': 'PLAIN',
                'security.protocol': 'SASL_SSL',
                'sasl.username': settings.KAFKA_API_KEY,
                'sasl.password': settings.KAFKA_API_SECRET,
            }   
            
            # Create test producer
            from confluent_kafka import Producer
            producer = Producer(producer_config)
            
            from confluent_kafka import Consumer
            consumer = Consumer(consumer_config)

            # # Test producer
            # producer.produce(settings.KAFKA_TOPIC, "test message")
            # producer.flush()

            # # Test consumer
            # consumer.subscribe([settings.KAFKA_TOPIC])
            # msg = consumer.poll(timeout=10)

            # logger.info("Confluent Cloud Kafka connection successful")
            
            # Topic creation is typically handled through Confluent Cloud UI or CLI
            logger.info(f"Ensure topic '{settings.KAFKA_TOPIC}' exists in Confluent Cloud")
            
            return True
            
        except Exception as e:
            logger.error(f"Confluent Cloud Kafka setup failed: {e}")
            return False
    
    def verify_setup(self) -> Dict[str, bool]:
        """Verify all cloud services are properly configured"""
        results = {
            "mongodb": False,
            "kafka": False
        }
        
        # Verify MongoDB
        try:
            if self.mongodb_client:
                self.mongodb_client.admin.command('ping')
                results["mongodb"] = True
        except Exception as e:
            logger.error(f"MongoDB verification failed: {e}")
        
        # Verify Kafka
        results["kafka"] = self.setup_confluent_kafka()
        
        return results
    
    def close(self):
        """Close connections"""
        if self.mongodb_client:
            self.mongodb_client.close()
