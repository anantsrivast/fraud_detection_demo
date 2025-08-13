#!/usr/bin/env python3
"""
Script to produce data to a specified Kafka topic.
Supports both transaction objects and generic JSON data.
"""

import json
import time
import random
import argparse
from models.transaction import Transaction
from services.cloud_kafka_service import CloudKafkaService
from config.settings import Settings

def generate_test_transaction(transaction_id: str) -> Transaction:
    """Generate a test transaction with realistic data"""
    
    # Sample data for realistic transactions
    merchants = [
        "Amazon.com", "Walmart", "Target", "Best Buy", "Home Depot",
        "Suspicious Merchant LLC", "Unknown Vendor", "Test Store"
    ]
    
    locations = [
        "New York, NY", "Los Angeles, CA", "Chicago, IL", "Houston, TX",
        "Phoenix, AZ", "Philadelphia, PA", "San Antonio, TX", "San Diego, CA"
    ]
    
    # Generate random amount (some high amounts to trigger fraud detection)
    amounts = [50, 100, 250, 500, 750, 1000, 1500, 2000, 5000]
    amount = random.choice(amounts)
    
    # Generate random customer ID
    customer_id = f"customer-{random.randint(1000, 9999)}"
    
    # Generate random merchant and location
    merchant = random.choice(merchants)
    location = random.choice(locations)
    
    # Generate timestamp (within last 24 hours)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    
    return Transaction(
        transaction_id=transaction_id,
        customer_id=customer_id,
        amount=amount,
        merchant=merchant,
        location=location,
        timestamp=timestamp
    )

def generate_fraud_alert(alert_id: str) -> dict:
    """Generate a fraud alert message"""
    return {
        "alert_id": alert_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "severity": random.choice(["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
        "fraud_type": random.choice(["SUSPICIOUS_AMOUNT", "UNUSUAL_LOCATION", "MULTIPLE_ATTEMPTS", "UNKNOWN_MERCHANT"]),
        "customer_id": f"customer-{random.randint(1000, 9999)}",
        "transaction_id": f"txn-{random.randint(100000, 999999)}",
        "risk_score": random.uniform(0.1, 1.0),
        "description": "Suspicious transaction detected by fraud detection system"
    }

def main():
    parser = argparse.ArgumentParser(description='Produce data to Kafka topic')
    parser.add_argument('--topic', required=True, help='Target Kafka topic name')
    parser.add_argument('--count', type=int, default=10, help='Number of messages to produce')
    parser.add_argument('--type', choices=['transaction', 'fraud_alert', 'custom'], 
                       default='transaction', help='Type of data to produce')
    parser.add_argument('--delay', type=float, default=1.0, help='Delay between messages in seconds')
    parser.add_argument('--custom-data', help='Custom JSON data to send (for custom type)')
    
    args = parser.parse_args()
    
    print(f"Starting Kafka Producer")
    print(f"Target topic: {args.topic}")
    print(f"Message count: {args.count}")
    print(f"Message type: {args.type}")
    print(f"Delay: {args.delay}s")
    print("=" * 50)
    
    # Check Kafka configuration
    if not all([Settings.KAFKA_BOOTSTRAP_SERVERS, 
                Settings.KAFKA_API_KEY, 
                Settings.KAFKA_API_SECRET]):
        print("ERROR: Kafka configuration incomplete. Please set:")
        print("   - KAFKA_BOOTSTRAP_SERVERS")
        print("   - KAFKA_API_KEY")
        print("   - KAFKA_API_SECRET")
        return
    
    # Initialize Kafka service
    kafka_service = CloudKafkaService()
    
    try:
        print(f"Connecting to Kafka cluster...")
        
        # Produce messages based on type
        for i in range(args.count):
            message_id = f"{args.type}-{i+1:03d}"
            
            if args.type == 'transaction':
                transaction = generate_test_transaction(message_id)
                success = kafka_service.publish_transaction(transaction, args.topic)
                if success:
                    print(f"Published transaction {i+1}/{args.count}: {transaction.transaction_id}")
                    print(f"   Amount: ${transaction.amount}")
                    print(f"   Merchant: {transaction.merchant}")
                else:
                    print(f"Failed to publish transaction {i+1}/{args.count}")
                    
            elif args.type == 'fraud_alert':
                alert_data = generate_fraud_alert(message_id)
                success = kafka_service.publish_message(message_id, alert_data, args.topic)
                if success:
                    print(f"Published fraud alert {i+1}/{args.count}: {alert_data['alert_id']}")
                    print(f"   Severity: {alert_data['severity']}")
                    print(f"   Risk Score: {alert_data['risk_score']:.2f}")
                else:
                    print(f"Failed to publish fraud alert {i+1}/{args.count}")
                    
            elif args.type == 'custom':
                if not args.custom_data:
                    print("ERROR: --custom-data is required for custom type")
                    return
                
                try:
                    custom_data = json.loads(args.custom_data)
                    success = kafka_service.publish_message(message_id, custom_data, args.topic)
                    if success:
                        print(f"Published custom message {i+1}/{args.count}: {message_id}")
                        print(f"   Data: {str(custom_data)[:100]}...")
                    else:
                        print(f"Failed to publish custom message {i+1}/{args.count}")
                except json.JSONDecodeError:
                    print("ERROR: Invalid JSON in --custom-data")
                    return
            
            # Delay between messages
            if i < args.count - 1:  # Don't delay after the last message
                time.sleep(args.delay)
        
        print("\n" + "=" * 50)
        print(f"Successfully produced {args.count} messages to topic: {args.topic}")
        print(f"Check your Kafka topic '{args.topic}' for the messages")
        
    except Exception as e:
        print(f"Error producing messages: {e}")
    finally:
        kafka_service.close()

if __name__ == "__main__":
    main() 