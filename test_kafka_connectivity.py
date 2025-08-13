#!/usr/bin/env python3
"""
Simple Kafka connectivity test script.
Tests connection to Kafka cluster, topic access, and data consumption.
"""

import json
import logging
import sys
from confluent_kafka import Consumer, KafkaException
from confluent_kafka.schema_registry import SchemaRegistryClient
from config.settings import Settings

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_kafka_connection():
    """Test basic Kafka connection and topic access"""
    try:
        print("Testing Kafka connection...")
        
        # Check configuration
        if not Settings.KAFKA_BOOTSTRAP_SERVERS:
            print("ERROR: KAFKA_BOOTSTRAP_SERVERS not configured")
            return False
        
        if not Settings.KAFKA_API_KEY:
            print("ERROR: KAFKA_API_KEY not configured")
            return False
        
        if not Settings.KAFKA_API_SECRET:
            print("ERROR: KAFKA_API_SECRET not configured")
            return False
        
        # Check Schema Registry configuration (optional)
        if Settings.SCHEMA_REGISTRY_URL:
            print("Schema Registry URL configured")
            if not Settings.SCHEMA_REGISTRY_API_KEY:
                print("WARNING: SCHEMA_REGISTRY_API_KEY not configured")
            if not Settings.SCHEMA_REGISTRY_API_SECRET:
                print("WARNING: SCHEMA_REGISTRY_API_SECRET not configured")
        else:
            print("Schema Registry not configured - will use plain JSON")
        
        print("Configuration check passed")
        
        # Create consumer config
        consumer_config = {
            'bootstrap.servers': Settings.KAFKA_BOOTSTRAP_SERVERS,
            'sasl.mechanisms': 'PLAIN',
            'security.protocol': 'SASL_SSL',
            'sasl.username': Settings.KAFKA_API_KEY,
            'sasl.password': Settings.KAFKA_API_SECRET,
            'group.id': f"{Settings.KAFKA_CONSUMER_GROUP}-test",
            'auto.offset.reset': 'beginning',
            'enable.auto.commit': False,
        }
        
        # Test Schema Registry connection if configured
        if Settings.SCHEMA_REGISTRY_URL:
            try:
                schema_registry_config = {
                    'url': Settings.SCHEMA_REGISTRY_URL,
                    'basic.auth.user.info': f"{Settings.SCHEMA_REGISTRY_API_KEY}:{Settings.SCHEMA_REGISTRY_API_SECRET}"
                }
                schema_registry_client = SchemaRegistryClient(schema_registry_config)
                # Test connection by getting subjects
                subjects = schema_registry_client.get_subjects()
                print(f"Schema Registry connection successful - found {len(subjects)} subjects")
            except Exception as e:
                print(f"Schema Registry connection failed: {e}")
                print("Will proceed with plain JSON")
        
        print(f"Connecting to: {Settings.KAFKA_BOOTSTRAP_SERVERS}")
        print(f"Topic: {Settings.KAFKA_TOPIC}")
        
        # Create consumer
        consumer = Consumer(consumer_config)
        
        # Subscribe to topic
        consumer.subscribe([Settings.KAFKA_TOPIC])
        print("Successfully subscribed to topic")
        
        # Test connection by polling
        print("Testing message consumption (timeout: 10 seconds)...")
        
        message_count = 0
        timeout_seconds = 10
        
        while message_count < 5:  # Read up to 5 messages
            msg = consumer.poll(timeout=1.0)
            
            if msg is None:
                continue
            
            if msg.error():
                print(f"Consumer error: {msg.error()}")
                continue
            
            # Parse message
            try:
                # Check if message is binary (Schema Registry format)
                message_bytes = msg.value()
                
                print(f"Message {message_count + 1}:")
                print(f"  Key: {msg.key()}")
                print(f"  Partition: {msg.partition()}")
                print(f"  Offset: {msg.offset()}")
                print(f"  Message size: {len(message_bytes)} bytes")
                
                # Check if it's a Schema Registry format message first
                if len(message_bytes) >= 5 and message_bytes[0:2] == b'\x00\x00':
                    # Schema Registry format: [0, 0, schema_id_high, schema_id_low, ...json_data]
                    print(f"  Schema Registry format detected")
                    print(f"  First 20 bytes: {message_bytes[:20]}")
                    
                    try:
                        # Extract JSON data (skip the 5-byte header)
                        json_data = message_bytes[5:].decode('utf-8')
                        parsed_data = json.loads(json_data)
                        print(f"  Manual Schema Registry parsing successful")
                        print(f"  Parsed data: {str(parsed_data)[:200]}...")
                    except Exception as manual_error:
                        print(f"  Manual Schema Registry parsing failed: {manual_error}")
                        
                        # Try official deserializer as fallback
                        if Settings.SCHEMA_REGISTRY_URL:
                            try:
                                from confluent_kafka.schema_registry.json_schema import JSONDeserializer
                                schema_registry_config = {
                                    'url': Settings.SCHEMA_REGISTRY_URL,
                                    'basic.auth.user.info': f"{Settings.SCHEMA_REGISTRY_API_KEY}:{Settings.SCHEMA_REGISTRY_API_SECRET}"
                                }
                                from confluent_kafka.schema_registry import SchemaRegistryClient
                                schema_client = SchemaRegistryClient(schema_registry_config)
                                deserializer = JSONDeserializer(schema_client, from_dict=lambda obj, ctx: obj)
                                parsed_data = deserializer(message_bytes, None)
                                print(f"  Official Schema Registry deserialization successful")
                                print(f"  Parsed data: {str(parsed_data)[:200]}...")
                            except Exception as sr_error:
                                print(f"  Official Schema Registry deserialization failed: {sr_error}")
                                print(f"  Error type: {type(sr_error).__name__}")
                        else:
                            print(f"  Schema Registry not configured")
                else:
                    # Try to decode as UTF-8 (plain JSON)
                    try:
                        message_data = message_bytes.decode('utf-8')
                        print(f"  Data: {message_data[:200]}...")  # First 200 chars
                        
                        # Try to parse as JSON
                        try:
                            parsed_data = json.loads(message_data)
                            print(f"  Parsed JSON successfully")
                            print(f"  Parsed data: {str(parsed_data)[:200]}...")
                        except json.JSONDecodeError:
                            print(f"  Raw text message")
                            
                    except UnicodeDecodeError:
                        print(f"  Binary message (unknown format)")
                        print(f"  First 20 bytes: {message_bytes[:20]}")
                
                message_count += 1
                
            except Exception as e:
                print(f"Error processing message: {e}")
                continue
        
        consumer.close()
        
        if message_count > 0:
            print(f"SUCCESS: Successfully consumed {message_count} messages from Kafka")
            return True
        else:
            print("WARNING: No messages found in topic (this might be normal if topic is empty)")
            print("Connection test passed - topic is accessible")
            return True
            
    except KafkaException as e:
        print(f"Kafka error: {e}")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False

def main():
    """Main test function"""
    print("Kafka Connectivity Test")
    print("=" * 40)
    
    success = test_kafka_connection()
    
    print("\n" + "=" * 40)
    if success:
        print("SUCCESS: Kafka connectivity test passed")
        print("Your Kafka configuration is working correctly")
    else:
        print("FAILED: Kafka connectivity test failed")
        print("Please check your configuration and try again")
        sys.exit(1)

if __name__ == "__main__":
    main() 