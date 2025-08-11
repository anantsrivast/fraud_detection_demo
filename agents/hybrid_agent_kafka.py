from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from models.transaction import Transaction
from services.fraud_signature_service import FraudSignatureService
from services.cloud_kafka_service import CloudKafkaService
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, model_validator
import json
import logging
import asyncio
from config.settings import Settings

logger = logging.getLogger(__name__)

# -------------------- State Definition --------------------

class WorkflowState(BaseModel):
    """
    Represents the state passed between nodes in the LangGraph workflow.

    Attributes:
        transaction: The incoming transaction data.
        duplicate_check: Output of duplicate detection step.
        fraud_analysis: Output of fraud classification.
        similarity_data: Output of vector similarity search.
        recommendation: LLM-generated action recommendation.
        storage_status: Result of storing the transaction.
        errors: List of error messages collected during processing.
    """
    transaction: Transaction
    duplicate_check: Optional[Dict[str, Any]] = None
    fraud_analysis: Optional[Dict[str, Any]] = None
    similarity_data: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    storage_status: Optional[str] = None
    errors: List[str] = []
    agent_reflection: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def convert_transaction_if_needed(cls, values):
        txn = values.get("transaction")
        if isinstance(txn, dict):
            values["transaction"] = Transaction(**txn)
        return values

# -------------------- MongoDB Connection --------------------

def get_db():
    """
    Establishes and returns a MongoDB database connection.
    """
    client = MongoClient(Settings.MONGODB_URI)
    return client[Settings.MONGODB_DATABASE]

# -------------------- Duplicate Check --------------------

async def duplicate_check(state: WorkflowState) -> WorkflowState:
    """
    Checks if the transaction is a duplicate by querying MongoDB.
    Adds a recommendation to skip or continue based on match.
    """
    try:
        transaction = state.transaction

        def check_duplicate():
            db = get_db()
            return db[Settings.MONGODB_COLLECTION_CASES].count_documents(
                {"transaction_id": transaction.transaction_id}
            ) > 0

        loop = asyncio.get_event_loop()
        is_duplicate = await loop.run_in_executor(None, check_duplicate)

        result = {
            "is_duplicate": is_duplicate,
            "customer_id": transaction.customer_id,
            "transaction_id": transaction.transaction_id,
            "checked_at": loop.time(),
            "recommendation": "SKIP_PROCESSING" if is_duplicate else "CONTINUE_PROCESSING"
        }

        if is_duplicate:
            result["reason"] = "Duplicate complaint detected within 24 hours"

        state.duplicate_check = result
        logger.info(f"Duplicate check completed for {transaction.transaction_id}: {is_duplicate}")

    except Exception as e:
        state.errors.append(f"duplicate_check error: {str(e)}")

    return state

# -------------------- Fraud Classification --------------------

async def fraud_classification(state: WorkflowState) -> WorkflowState:
    """
    Classifies the transaction as fraud or not based on the amount.
    """
    try:
        amt = state.transaction.amount or 0
        state.fraud_analysis = {"is_fraud": amt > 1000}
    except Exception as e:
        state.errors.append(f"fraud_classification error: {str(e)}")
    return state

# -------------------- Similarity Search --------------------

async def similarity_search(state: WorkflowState) -> WorkflowState:
    """
    Searches for similar cases in the database using vector similarity.
    """
    try:
        loop = asyncio.get_event_loop()
        encoder = SentenceTransformer(Settings.EMBEDDING_MODEL)
        fs = FraudSignatureService()
        signatures = fs.generate_fraud_signatures(state.transaction)

        def encode():
            return encoder.encode([" ".join(signatures)])[0]

        embedding = await loop.run_in_executor(None, encode)

        def search():
            db = get_db()
            return list(db[Settings.MONGODB_COLLECTION_SIGNATURES].aggregate([
                {"$vectorSearch": {
                    "index": Settings.VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding,
                    "numCandidates": 50,
                    "limit": 5,
                }}
            ]))

        results = await loop.run_in_executor(None, search)
        state.similarity_data = {"similar_cases": results}

        def insert_signature():
            db = get_db()
            db[Settings.MONGODB_COLLECTION_SIGNATURES].insert_one({
                "transaction_id": state.transaction.transaction_id,
                "signatures": signatures,
                "embedding": embedding
            })

        if state.fraud_analysis and state.fraud_analysis.get("is_fraud"):
            await loop.run_in_executor(None, insert_signature)

    except Exception as e:
        state.errors.append(f"similarity_search error: {str(e)}")
    return state

async def agent_reflection(state: WorkflowState) -> WorkflowState:
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        prompt = f"""
        As an AI fraud analyst, reflect on the following:

        - Fraud analysis result: {state.fraud_analysis}
        - Similarity search result: {state.similarity_data}

        What patterns stand out? What initial actions would you take and why?
        """
        response = await llm.ainvoke(prompt)
        state.agent_reflection = str(response.content if hasattr(response, "content") else response)

    except Exception as e:
        state.errors.append(f"agent_reflection error: {str(e)}")
    return state

async def action_recommendation(state: WorkflowState) -> WorkflowState:
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        context = f"Fraud: {state.fraud_analysis}\nSimilar: {state.similarity_data}"
        prompt = f"""
        Given the analysis:
        {context}

        What should the fraud response team do?
        Explain your reasoning step-by-step before making a recommendation.
        """
        response = await llm.ainvoke(prompt)
        state.recommendation = str(response.content if hasattr(response, "content") else response)

    except Exception as e:
        state.errors.append(f"action_recommendation error: {str(e)}")
    return state

async def store_transaction(state: WorkflowState) -> WorkflowState:
    try:
        def store():
            db = get_db()
            db[Settings.MONGODB_COLLECTION_CASES].insert_one(state.transaction.dict())

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store)
        state.storage_status = "stored"
    except Exception as e:
        state.errors.append(f"storage error: {str(e)}")
    return state



def build_workflow():
    workflow = StateGraph(WorkflowState)

    workflow.add_node("duplicate", duplicate_check)
    workflow.add_node("classify", fraud_classification)
    workflow.add_node("similarity", similarity_search)
    workflow.add_node("reflect", agent_reflection)
    workflow.add_node("recommend", action_recommendation)
    workflow.add_node("store", store_transaction)

    workflow.set_entry_point("duplicate")
    workflow.add_conditional_edges(
        "duplicate",
        lambda s: "skip" if s.duplicate_check and s.duplicate_check.get("is_duplicate") else "classify",
        {"skip": END, "classify": "classify"}
    )
    workflow.add_conditional_edges(
        "classify",
        lambda s: "similarity" if s.fraud_analysis and s.fraud_analysis.get("is_fraud") else "skip",
        {"similarity": "similarity", "skip": END}
    )
    workflow.add_edge("similarity", "reflect")
    workflow.add_edge("reflect", "recommend")
    workflow.add_edge("recommend", "store")
    workflow.add_edge("store", END)

    return workflow.compile()

async def process_transaction_from_kafka(transaction: Transaction):
    """Process a single transaction from Kafka through the workflow"""
    try:
        print(f"\n========= Processing Transaction: {transaction.transaction_id} =========")
        state = WorkflowState(transaction=transaction)
        final_state = await compiled.ainvoke(state)
        print(f"Final state for {transaction.transaction_id}: {final_state}")
        return final_state
    except Exception as e:
        logger.error(f"Error processing transaction {transaction.transaction_id}: {e}")
        return None

def kafka_transaction_callback(transaction: Transaction):
    """Callback function for Kafka consumer to process transactions"""
    try:
        # Create a new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async workflow
        if loop.is_running():
            # If we're already in an event loop, create a task
            asyncio.create_task(process_transaction_from_kafka(transaction))
        else:
            # If no event loop is running, run the coroutine
            loop.run_until_complete(process_transaction_from_kafka(transaction))
    except Exception as e:
        logger.error(f"Error in Kafka callback: {e}")

if __name__ == "__main__":
    compiled = build_workflow()
    
    # Initialize Kafka service
    kafka_service = CloudKafkaService()
    
    try:
        print("Starting Kafka consumer for fraud detection workflow...")
        print(f"Listening on topic: {Settings.KAFKA_TOPIC}")
        print("Press Ctrl+C to stop...")
        
        # Start consuming transactions from Kafka
        kafka_service.consume_transactions(kafka_transaction_callback)
        
    except KeyboardInterrupt:
        print("\nShutting down...")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
    finally:
        kafka_service.close()
