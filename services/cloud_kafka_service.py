from confluent_kafka import Producer, Consumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.json_schema import JSONSerializer, JSONDeserializer
import json
import logging
import time
from typing import Callable, Optional
from models.transaction import Transaction
from config.settings import Settings

logger = logging.getLogger(__name__)

class CloudKafkaService:
    def __init__(self):
        self.producer_config = {
            'bootstrap.servers': Settings.KAFKA_BOOTSTRAP_SERVERS,
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': Settings.KAFKA_API_KEY,
            'sasl.password': Settings.KAFKA_API_SECRET,
            'acks': 'all',
            'retries': 3,
        }
        
        self.consumer_config = {
            'bootstrap.servers': Settings.KAFKA_BOOTSTRAP_SERVERS,
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': Settings.KAFKA_API_KEY,
            'sasl.password': Settings.KAFKA_API_SECRET,
            'group.id': Settings.KAFKA_CONSUMER_GROUP,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
        }

        # Schema Registry configuration
        self.schema_registry_config = {
            'url': Settings.SCHEMA_REGISTRY_URL,
            'basic.auth.user.info': f"{Settings.SCHEMA_REGISTRY_API_KEY}:{Settings.SCHEMA_REGISTRY_API_SECRET}"
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
            logger.info("🔧 Creating Kafka consumer...")
            logger.info(f"   Consumer config: {self.consumer_config}")
            
            self.consumer = Consumer(self.consumer_config)
            logger.info("Consumer object created")
            
            # Use manual partition assignment instead of group-based assignment
            # This ensures we get assigned to partitions immediately
            from confluent_kafka import TopicPartition
            
            # Get topic metadata to find available partitions
            logger.info(f"🔍 Getting topic metadata for: {Settings.KAFKA_TOPIC}")
            metadata = self.consumer.list_topics(Settings.KAFKA_TOPIC)
            topic_metadata = metadata.topics.get(Settings.KAFKA_TOPIC)
            
            if topic_metadata and topic_metadata.partitions:
                logger.info(f"Found topic with {len(topic_metadata.partitions)} partitions")
                # Assign to all available partitions
                partitions = [TopicPartition(Settings.KAFKA_TOPIC, p) for p in topic_metadata.partitions.keys()]
                logger.info(f"   Partitions to assign: {[p.partition for p in partitions]}")
                self.consumer.assign(partitions)
                logger.info(f"Confluent Kafka consumer manually assigned to {len(partitions)} partitions for topic: {Settings.KAFKA_TOPIC}")
            else:
                logger.warning(f"Topic not found or no partitions, falling back to group-based assignment")
                # Fallback to group-based assignment
                self.consumer.subscribe([Settings.KAFKA_TOPIC])
                logger.info(f"Confluent Kafka consumer using group-based assignment for topic: {Settings.KAFKA_TOPIC}")
            
            return self.consumer
        except Exception as e:
            logger.error(f"Failed to create Confluent Kafka consumer: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
            raise
    
    def setup_schema_registry(self):
        """Setup Schema Registry client and serializers"""
        # For now, disable Schema Registry setup to avoid deserializer issues
        # We'll rely on manual parsing of Schema Registry format messages
        logger.info("Schema Registry setup disabled - using manual parsing")
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
            target_topic = topic or Settings.KAFKA_OUTPUT_TOPIC

            # Prepare transaction data with datetime handling
            transaction_dict = transaction.model_dump()
            # Convert datetime objects to ISO format strings
            for key, value in transaction_dict.items():
                if hasattr(value, 'isoformat'):
                    transaction_dict[key] = value.isoformat()

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
                logger.info(f"Published fraud result to {Settings.KAFKA_OUTPUT_TOPIC} using Schema Registry")
            else:
                serialized_data = json.dumps(result_data).encode('utf-8')
                logger.info(f"Published fraud result to {Settings.KAFKA_OUTPUT_TOPIC} using plain JSON")

            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f'Message delivery failed: {err}')
                else:
                    logger.info(f'Fraud result delivered to {msg.topic()} [{msg.partition()}]')

            self.producer.produce(
                Settings.KAFKA_OUTPUT_TOPIC,
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
                
            logger.info(f"Starting to consume transactions from topic: {Settings.KAFKA_TOPIC}")
            logger.info(f"Consumer group: {Settings.KAFKA_CONSUMER_GROUP}")
            logger.info(f"Auto offset reset: earliest")
            
            # Get consumer assignment to verify subscription
            assignment = self.consumer.assignment()
            logger.info(f"Consumer assignment: {assignment}")
            
            if not assignment:
                logger.error("No partition assignment - consumer cannot receive messages!")
                logger.error("This could be due to:")
                logger.error("  - No messages in the topic")
                logger.error("  - Consumer group issues")
                logger.error("  - Topic permissions")
                return
            
            logger.info(f"✅ Consumer assigned to {len(assignment)} partitions")
            
            # Show partition details
            for partition in assignment:
                logger.info(f"  Partition {partition.partition}: offset {partition.offset}")
            
            poll_count = 0
            while True:
                try:
                    poll_count += 1
                    logger.info(f"Poll #{poll_count} - waiting for message...")
                    
                    msg = self.consumer.poll(timeout=1.0)
                    
                    if msg is None:
                        logger.info(f"   Poll #{poll_count}: No message received")
                        continue
                    
                    if msg.error():
                        logger.error(f"   Poll #{poll_count}: Consumer error: {msg.error()}")
                        continue
                    
                    logger.info(f"🎉 Poll #{poll_count}: MESSAGE RECEIVED!")
                    logger.info(f"   Topic: {msg.topic()}")
                    logger.info(f"   Partition: {msg.partition()}")
                    logger.info(f"   Offset: {msg.offset()}")
                    logger.info(f"   Key: {msg.key()}")
                    logger.info(f"   Message size: {len(msg.value())} bytes")
                    
                    # Parse message 
                    message_bytes = msg.value()
                    logger.info(f"   Message bytes: {message_bytes[:20]}... (first 20 bytes)")

                    # Parse message using manual Schema Registry format detection
                    # Check if it's a Schema Registry format message
                    if len(message_bytes) >= 5 and message_bytes[0:2] == b'\x00\x00':
                        logger.info("   Detected Schema Registry format message")
                        # Schema Registry format: [0, 0, schema_id_high, schema_id_low, ...json_data]
                        try:
                            # Extract JSON data (skip the 5-byte header)
                            json_data = message_bytes[5:].decode('utf-8')
                            logger.info(f"   Extracted JSON data: {json_data[:100]}... (first 100 chars)")
                            transaction_dict = json.loads(json_data)
                            logger.info("Successfully parsed Schema Registry message manually")
                        except Exception as manual_error:
                            logger.error(f"Manual Schema Registry parsing failed: {manual_error}")
                            continue
                    else:
                        logger.info("   Detected plain JSON format message")
                        # Try plain JSON deserialization
                        try:
                            transaction_dict = json.loads(msg.value().decode('utf-8'))
                            logger.info("Deserialized message using plain JSON")
                        except UnicodeDecodeError as decode_error:
                            logger.error(f"Failed to decode message as UTF-8: {decode_error}")
                            logger.error("   Message appears to be binary but not Schema Registry format")
                            continue
                        except json.JSONDecodeError as json_error:
                            logger.error(f"Failed to parse JSON: {json_error}")
                            continue

                    logger.info(f"   Transaction dict keys: {list(transaction_dict.keys())}")
                    
                    try:
                        transaction = Transaction(**transaction_dict)
                        logger.info(f"Successfully created Transaction object: {transaction.transaction_id}")
                        callback(transaction)
                        logger.info(f"Callback executed successfully")
                    except Exception as transaction_error:
                        logger.error(f"Failed to create Transaction object: {transaction_error}")
                        logger.error(f"   Transaction dict: {transaction_dict}")
                        continue
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Poll #{poll_count}: Failed to decode message: {e}")
                    continue
                except Exception as e:
                    logger.error(f"Poll #{poll_count}: Error processing message: {e}")
                    import traceback
                    logger.error(f"   Traceback: {traceback.format_exc()}")
                    continue
                    
        except KeyboardInterrupt:
            logger.info("Consumer stopped by user")
        except Exception as e:
            logger.error(f"Error in consumer: {e}")
            import traceback
            logger.error(f"   Traceback: {traceback.format_exc()}")
        finally:
            if self.consumer:
                self.consumer.close()
                logger.info("Consumer closed")
    
    def close(self):
        """Close producer and consumer"""
        if self.producer:
            self.producer.flush()
        if self.consumer:
            self.consumer.close()
        logger.info("Confluent Kafka service closed")
