from confluent_kafka import Producer, Consumer, KafkaException
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.json_schema import JSONSerializer, JSONDeserializer
import json
import logging
import time
from typing import Callable, Optional, Dict, Any
from models.transaction import Transaction
from config.settings import settings

logger = logging.getLogger(__name__)

class CloudKafkaService:
    def __init__(self):
        self.producer_config = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': settings.KAFKA_API_KEY,
            'sasl.password': settings.KAFKA_API_SECRET,
            'acks': 'all',
            'retries': 3,
        }
        
        self.consumer_config = {
            'bootstrap.servers': settings.KAFKA_BOOTSTRAP_SERVERS,
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': settings.KAFKA_API_KEY,
            'sasl.password': settings.KAFKA_API_SECRET,
            'group.id': settings.KAFKA_CONSUMER_GROUP,
            'auto.offset.reset': 'beginning',
            'enable.auto.commit': True,
        }

        # Schema Registry configuration
        self.schema_registry_config = {
            'url': settings.SCHEMA_REGISTRY_URL,
            'basic.auth.user.info': f"{settings.SCHEMA_REGISTRY_API_KEY}:{settings.SCHEMA_REGISTRY_API_SECRET}"
        }
        
        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None
        self.schema_registry_client: Optional[SchemaRegistryClient] = None
        self.json_serializer: Optional[JSONSerializer] = None
        self.json_deserializer: Optional[JSONDeserializer] = None
    
    def create_producer(self) -> Producer:
        """Create Confluent Kafka producer"""
        try:
            self.producer = Producer(self.producer_config)
            logger.info("Confluent Kafka producer created successfully")
            return self.producer
        except Exception as e:
            logger.error(f"Failed to create Confluent Kafka producer: {e}")
            raise
    
    def create_consumer(self) -> Consumer:
        """Create Confluent Kafka consumer"""
        try:
            self.consumer = Consumer(self.consumer_config)
            self.consumer.subscribe([settings.KAFKA_TOPIC])
            logger.info(f"Confluent Kafka consumer created for topic: {settings.KAFKA_TOPIC}")
            return self.consumer
        except Exception as e:
            logger.error(f"Failed to create Confluent Kafka consumer: {e}")
            raise
    
    def setup_schema_registry(self):
        """Setup Schema Registry client and serializers"""
        try:
            if not settings.SCHEMA_REGISTRY_URL:
                logger.warning("Schema Registry URL not configured, using plain JSON")
                return

            self.schema_registry_client = SchemaRegistryClient(self.schema_registry_config)

            # Test Schema Registry connection
            try:
                subjects = self.schema_registry_client.get_subjects()
                logger.info(f"Schema Registry connection successful - found {len(subjects)} subjects")
            except Exception as e:
                logger.error(f"Schema Registry connection test failed: {e}")
                raise

            # Create JSON serializer for producing
            self.json_serializer = JSONSerializer(
                self.schema_registry_client,
                settings.SCHEMA_REGISTRY_SUBJECT
            )

            # Create JSON deserializer for consuming
            # For deserialization, we don't specify a subject - it auto-detects from the message
            # The from_dict parameter tells it how to convert the parsed JSON back to Python objects
            self.json_deserializer = JSONDeserializer(
                self.schema_registry_client,
                from_dict=lambda obj, ctx: obj,
                to_dict=lambda obj, ctx: obj
            )

            logger.info("Schema Registry setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to setup Schema Registry: {e}")
            # Fall back to plain JSON
            self.schema_registry_client = None
            self.json_serializer = None
            self.json_deserializer = None

    def publish_transaction(self, transaction: Transaction, topic: str = None) -> bool:
        """Publish transaction to Confluent Kafka topic with Schema Registry"""
        try:
            if not self.producer:
                self.create_producer()
            
                        # Setup schema registry if not already done
            if not self.schema_registry_client:
                self.setup_schema_registry()

            # Use specified topic or default to output topic
            target_topic = topic or settings.KAFKA_OUTPUT_TOPIC

            # Prepare transaction data
            transaction_dict = transaction.model_dump(default=str)

            # Use schema registry serializer if available, otherwise plain JSON
            if self.json_serializer:
                serialized_data = self.json_serializer(transaction_dict, None)
                logger.info(f"Published transaction to {target_topic} using Schema Registry")
            else:
                serialized_data = json.dumps(transaction_dict).encode('utf-8')
                logger.info(f"Published transaction to {target_topic} using plain JSON")
            
            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f'Message delivery failed: {err}')
                else:
                    logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
            
            self.producer.produce(
                target_topic,
                key=transaction.transaction_id,
                value=serialized_data,
                callback=delivery_report
            )
            
            # Wait for message to be delivered
            self.producer.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish transaction: {e}")
            return False
    def publish_message(self, key: str, data: dict, topic: str) -> bool:
        """Publish any message to a specified Kafka topic"""
        try:
            if not self.producer:
                self.create_producer()

            # Setup schema registry if not already done
            if not self.schema_registry_client:
                self.setup_schema_registry()

            # Use schema registry serializer if available, otherwise plain JSON
            if self.json_serializer:
                serialized_data = self.json_serializer(data, None)
                logger.info(f"Published message to {topic} using Schema Registry")
            else:
                serialized_data = json.dumps(data).encode('utf-8')
                logger.info(f"Published message to {topic} using plain JSON")

            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f'Message delivery failed: {err}')
                else:
                    logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')

            self.producer.produce(
                topic,
                key=key,
                value=serialized_data,
                callback=delivery_report
            )

            # Wait for message to be delivered
            self.producer.flush()
            return True

        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    def publish_fraud_result(self, transaction_id: str, fraud_result: dict) -> bool:
        """Publish fraud detection result to the output topic"""
        try:
            if not self.producer:
                self.create_producer()

            # Setup schema registry if not already done
            if not self.schema_registry_client:
                self.setup_schema_registry()

            # Prepare fraud result data
            result_data = {
                "transaction_id": transaction_id,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "fraud_result": fraud_result
            }

            # Use schema registry serializer if available, otherwise plain JSON
            if self.json_serializer:
                serialized_data = self.json_serializer(result_data, None)
                logger.info(f"Published fraud result to {settings.KAFKA_OUTPUT_TOPIC} using Schema Registry")
            else:
                serialized_data = json.dumps(result_data).encode('utf-8')
                logger.info(f"Published fraud result to {settings.KAFKA_OUTPUT_TOPIC} using plain JSON")

            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f'Message delivery failed: {err}')
                else:
                    logger.info(f'Fraud result delivered to {msg.topic()} [{msg.partition()}]')

            self.producer.produce(
                settings.KAFKA_OUTPUT_TOPIC,
                key=transaction_id,
                value=serialized_data,
                callback=delivery_report
            )

            # Wait for message to be delivered
            self.producer.flush()
            return True

        except Exception as e:
            logger.error(f"Failed to publish fraud result: {e}")
            return False
            
    def consume_transactions(self, callback: Callable[[Transaction], None]):
        """Consume transactions from Confluent Kafka topic"""
        try:
            if not self.consumer:
                self.create_consumer()
            
            # Setup schema registry if not already done
            if not self.schema_registry_client:
                self.setup_schema_registry()
                
            logger.info(f"Starting to consume transactions from topic: {settings.KAFKA_TOPIC}")
            
            while True:
                try:
                    msg = self.consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        logger.error(f"Consumer error: {msg.error()}")
                        continue
                    
                    # Parse message 
                    message_bytes = msg.value()

                    # Check if it's a Schema Registry format message
                    if len(message_bytes) >= 5 and message_bytes[0:2] == b'\x00\x00':
                        # Schema Registry format: [0, 0, schema_id_high, schema_id_low, ...json_data]
                        try:
                            # Extract JSON data (skip the 5-byte header)
                            json_data = message_bytes[5:].decode('utf-8')
                            transaction_dict = json.loads(json_data)
                            logger.info("Successfully parsed Schema Registry message")
                        except Exception as manual_error:
                            logger.error(f"Manual Schema Registry parsing failed: {manual_error}")
                            # Try official deserializer as fallback
                            if self.json_deserializer:
                                try:
                                    transaction_dict = self.json_deserializer(msg.value(), None)
                                    logger.info("Used official Schema Registry deserializer")
                                except Exception as e:
                                    logger.error(f"Official Schema Registry deserialization failed: {e}")
                                    continue
                            else:
                                continue
                    else:
                        # Try plain JSON deserialization
                        try:
                            transaction_dict = json.loads(msg.value().decode('utf-8'))
                            logger.info("Deserialized message using plain JSON")
                        except UnicodeDecodeError as decode_error:
                            logger.error(f"Failed to decode message as UTF-8: {decode_error}")
                            logger.error("Message appears to be binary but not Schema Registry format")
                            continue

                    transaction = Transaction(**transaction_dict)
                    
                    logger.info(f"Consumed transaction: {transaction.transaction_id}")
                    callback(transaction)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to decode message: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Error processing message: {e}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
        finally:
            if self.consumer:
                self.consumer.close()
    
    def close(self):
        """Close producer and consumer"""
        if self.producer:
            self.producer.flush()
        if self.consumer:
            self.consumer.close()
        logger.info("Confluent Kafka service closed")
