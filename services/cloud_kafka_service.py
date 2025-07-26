from confluent_kafka import Producer, Consumer, KafkaException
import json
import logging
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
            'auto.offset.reset': 'latest',
            'enable.auto.commit': True,
        }
        
        self.producer: Optional[Producer] = None
        self.consumer: Optional[Consumer] = None
    
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
    
    def publish_transaction(self, transaction: Transaction) -> bool:
        """Publish transaction to Confluent Kafka topic"""
        try:
            if not self.producer:
                self.create_producer()
            
            transaction_data = json.dumps(transaction.model_dump(default=str))
            
            def delivery_report(err, msg):
                if err is not None:
                    logger.error(f'Message delivery failed: {err}')
                else:
                    logger.info(f'Message delivered to {msg.topic()} [{msg.partition()}]')
            
            self.producer.produce(
                settings.KAFKA_TOPIC,
                key=transaction.transaction_id,
                value=transaction_data,
                callback=delivery_report
            )
            
            # Wait for message to be delivered
            self.producer.flush()
            return True
            
        except Exception as e:
            logger.error(f"Failed to publish transaction: {e}")
            return False
    
    def consume_transactions(self, callback: Callable[[Transaction], None]):
        """Consume transactions from Confluent Kafka topic"""
        try:
            if not self.consumer:
                self.create_consumer()
            
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
                    transaction_data = json.loads(msg.value().decode('utf-8'))
                    transaction = Transaction(**transaction_data)
                    
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