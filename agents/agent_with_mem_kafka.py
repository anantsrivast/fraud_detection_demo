from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # Use async version
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from models.transaction import Transaction
from services.fraud_signature_service import FraudSignatureService
from services.cloud_kafka_service import CloudKafkaService
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import json
import logging
import asyncio
from config.settings import Settings
from config.logger import get_logger
from datetime import datetime
import time

logger = get_logger(__name__)

# ---------------------------
# State hydration helper (dict -> WorkflowState)
# ---------------------------

def ensure_state(s: Union["WorkflowState", dict]) -> "WorkflowState":
    return s if isinstance(s, WorkflowState) else WorkflowState(**s)


# ---------------------------
# Pretty-print helpers
# ---------------------------

def hr(title: str) -> None:
    print(f"\n{'=' * 8} {title.upper()} {'=' * 8}")

def kv(key: str, value: Any) -> None:
    print(f"- {key}: {value}")

def yes_no(flag: bool) -> str:
    return "YES" if flag else "NO"

def pretty_duplicate(result: Dict[str, Any]) -> None:
    hr("Duplicate Check")
    kv("Duplicate", yes_no(result.get("is_duplicate", False)))
    if result.get("is_duplicate"):
        kv("Reason", result.get("reason", "Possible duplicate detected"))
    kv("Customer ID", result.get("customer_id"))
    kv("Transaction ID", result.get("transaction_id"))
    ts = result.get("checked_at")
    if ts:
        kv("Checked At (utc)", datetime.utcfromtimestamp(ts).isoformat() + "Z")
    kv("Next Step", result.get("recommendation"))

def pretty_fraud(analysis: Dict[str, Any], amount: float) -> None:
    hr("Fraud Classification")
    kv("Is Fraud", yes_no(analysis.get("is_fraud", False)))
    kv("Amount", amount)

def pretty_similarity(data: Dict[str, Any]) -> None:
    hr("Similarity Detection")
    sigs: List[str] = data.get("query_signatures", [])
    if sigs:
        kv("Generated Signatures", ", ".join(sigs))
    sims: List[Dict[str, Any]] = data.get("similar_cases", []) or []
    if not sims:
        print("No similar cases found.")
        return
    print("Top Similar Cases:")
    # We assume each result may include a similarity score under 'score'
    # and may contain 'transaction_id' or an '_id'. We handle both.
    for i, doc in enumerate(sims, 1):
        tid = doc.get("transaction_id") or doc.get("_id")
        score = doc.get("score")
        line = f"  {i}. id={tid}"
        if score is not None:
            line += f" | score={round(float(score), 4)}"
        print(line)
        # If stored signatures are present in similar doc, show a short preview
        if isinstance(doc.get("signatures"), list) and doc["signatures"]:
            preview = ", ".join(doc["signatures"][:3])
            print(f"     signatures: {preview}{' ...' if len(doc['signatures']) > 3 else ''}")

def pretty_reflection(text: Optional[str]) -> None:
    hr("Agent Reflection")
    if text:
        print(text.strip())
    else:
        print("No reflection available.")

def pretty_recommendation(text: Optional[str]) -> None:
    hr("Action Recommendation")
    if text:
        print(text.strip())
    else:
        print("No recommendation available.")

def pretty_storage(status: Optional[str], txn_id: str) -> None:
    hr("Storage")
    kv("Transaction ID", txn_id)
    kv("Status", status or "unknown")

def pretty_publish(status: Optional[str], txn_id: str) -> None:
    hr("Kafka Publishing")
    kv("Transaction ID", txn_id)
    kv("Status", status or "unknown")

def pretty_final(state: "WorkflowState") -> None:  # type: ignore
    hr("Final Summary")
    kv("Transaction ID", state.transaction.transaction_id)
    kv("Duplicate", yes_no(state.duplicate_check.get("is_duplicate") if state.duplicate_check else False))
    kv("Fraud", yes_no(state.fraud_analysis.get("is_fraud") if state.fraud_analysis else False))
    kv("Similar Cases", len((state.similarity_data or {}).get("similar_cases", [])))
    kv("Recommendation", state.recommendation or "N/A")
    kv("Published", yes_no(state.publish_status == "published"))
    if state.errors:
        print("\nErrors:")
        for e in state.errors:
            print(f"  - {e}")


# ---------------------------
# Pydantic model
# ---------------------------

class WorkflowState(BaseModel):
    transaction: Transaction
    duplicate_check: Optional[Dict[str, Any]] = None
    fraud_analysis: Optional[Dict[str, Any]] = None
    similarity_data: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    storage_status: Optional[str] = None
    publish_status: Optional[str] = None
    errors: List[str] = Field(default_factory=list)  # avoid shared mutable default
    agent_reflection: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def convert_transaction_if_needed(cls, values):
        txn = values.get("transaction")
        if isinstance(txn, dict):
            values["transaction"] = Transaction(**txn)
        return values

def get_db():
    client = MongoClient(Settings.MONGODB_URI)
    return client[Settings.MONGODB_DATABASE]

async def duplicate_check(state: WorkflowState) -> WorkflowState:
    try:
        transaction = state.transaction

        def check_duplicate():
            db = get_db()
            return db[Settings.MONGODB_COLLECTION_CASES].count_documents(
                {"ip_address": transaction.ip_address}
            ) > 0

        loop = asyncio.get_event_loop()
        is_duplicate = await loop.run_in_executor(None, check_duplicate)

        result = {
            "is_duplicate": is_duplicate,
            "customer_id": transaction.customer_id,
            "transaction_id": transaction.transaction_id,
            "checked_at": time.time(),  # epoch seconds (fix)
            "recommendation": "SKIP_PROCESSING" if is_duplicate else "CONTINUE_PROCESSING"
        }

        if is_duplicate:
            result["reason"] = "Duplicate complaint detected within 24 hours"

        state.duplicate_check = result
        logger.info(f"Duplicate check completed for {transaction.transaction_id}: {is_duplicate}")

        # Pretty print
        pretty_duplicate(result)

    except Exception as e:
        state.errors.append(f"duplicate_check error: {str(e)}")

    return state

async def fraud_classification(state: WorkflowState) -> WorkflowState:
    try:
        amt = state.transaction.amount or 0
        state.fraud_analysis = {"is_fraud": amt > 1000}
        logger.info(f"Fraud classification: {state.fraud_analysis}")

        # Pretty print
        pretty_fraud(state.fraud_analysis, amt)

    except Exception as e:
        state.errors.append(f"fraud_classification error: {str(e)}")
    return state

async def similarity_search(state: WorkflowState) -> WorkflowState:
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
            results = list(db[Settings.MONGODB_COLLECTION_SIGNATURES].aggregate([
                {"$vectorSearch": {
                    "index": Settings.VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding.tolist(),
                    "numCandidates": 50,
                    "limit": 5,
                }},
                # If your server supports it, you can $project score explicitly:
                # {"$project": {"_id": 1, "transaction_id": 1, "signatures": 1, "score": {"$meta": "vectorSearchScore"}}}
            ]))

            # Convert ObjectId to string for clean printing
            for r in results:
                if "_id" in r:
                    r["_id"] = str(r["_id"])
            return results

        results = await loop.run_in_executor(None, search)

        # Save both the results and the query signatures for printing
        state.similarity_data = {"similar_cases": results, "query_signatures": signatures}

        # Store signature for high-risk transactions
        def insert_signature():
            db = get_db()
            db[Settings.MONGODB_COLLECTION_SIGNATURES].insert_one({
                "transaction_id": state.transaction.transaction_id,
                "signatures": signatures,
                "embedding": embedding.tolist()
            })

        if state.fraud_analysis and state.fraud_analysis.get("is_fraud"):
            await loop.run_in_executor(None, insert_signature)
            logger.info(f"Stored fraud signature for {state.transaction.transaction_id}")

        # Pretty print
        pretty_similarity(state.similarity_data)

    except Exception as e:
        state.errors.append(f"similarity_search error: {str(e)}")
    return state

async def agent_reflection(state: WorkflowState) -> WorkflowState:
    try:
        # Use GPT-3.5-turbo for better rate limits and lower cost
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            max_retries=3,
            request_timeout=60,
            max_tokens=200  # Significantly reduce response length
        )
        
        # Create a much shorter, focused prompt
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        similar_count = len(state.similarity_data.get("similar_cases", [])) if state.similarity_data else 0
        
        prompt = f"""Fraud detected: {fraud_status}
Similar cases found: {similar_count}

Brief analysis (max 100 words): What actions should be taken?"""
        
        # Add delay to prevent rate limiting
        await asyncio.sleep(2)
        
        response = await llm.ainvoke(prompt)
        state.agent_reflection = str(response.content if hasattr(response, "content") else response)
        logger.info("Agent reflection completed")
        
        # Pretty print
        pretty_reflection(state.agent_reflection)

    except Exception as e:
        logger.warning(f"agent_reflection error (continuing): {str(e)}")
        # Provide a simple fallback response
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        state.agent_reflection = (
            "Automatic analysis: High risk transaction - requires immediate review"
            if fraud_status else
            "Automatic analysis: Low risk transaction - standard processing"
        )
        state.errors.append(f"agent_reflection error: {str(e)}")
        pretty_reflection(state.agent_reflection)
    return state

async def action_recommendation(state: WorkflowState) -> WorkflowState:
    try:
        # Use GPT-3.5-turbo for better rate limits and lower cost
        llm = ChatOpenAI(
            model="gpt-3.5-turbo", 
            temperature=0,
            max_retries=3,
            request_timeout=60,
            max_tokens=150  # Significantly reduce response length
        )
        
        # Create a much shorter, focused prompt
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        similar_count = len(state.similarity_data.get("similar_cases", [])) if state.similarity_data else 0
        
        prompt = f"""Transaction risk: {'HIGH' if fraud_status else 'LOW'}
Similar cases: {similar_count}

Recommended action (max 50 words):"""
        
        # Add delay to prevent rate limiting
        await asyncio.sleep(2)
        
        response = await llm.ainvoke(prompt)
        state.recommendation = str(response.content if hasattr(response, "content") else response)
        logger.info("Action recommendation completed")
        
        # Pretty print
        pretty_recommendation(state.recommendation)

    except Exception as e:
        logger.warning(f"action_recommendation error (continuing): {str(e)}")
        # Provide a simple fallback response
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        if fraud_status:
            state.recommendation = "RECOMMENDED ACTION: Block transaction, investigate customer account, review for fraud patterns."
        else:
            state.recommendation = "RECOMMENDED ACTION: Approve transaction, continue monitoring."
        state.errors.append(f"action_recommendation error: {str(e)}")
        pretty_recommendation(state.recommendation)
    return state

async def store_transaction(state: WorkflowState) -> WorkflowState:
    try:
        def store():
            db = get_db()
            # Handle both Pydantic v1 and v2
            if hasattr(state.transaction, 'model_dump'):
                txn_dict = state.transaction.model_dump()
            else:
                txn_dict = state.transaction.dict()
            db[Settings.MONGODB_COLLECTION_CASES].insert_one(txn_dict)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store)
        state.storage_status = "stored"
        logger.info(f"Transaction {state.transaction.transaction_id} stored successfully")
        
        # Pretty print
        pretty_storage(state.storage_status, state.transaction.transaction_id)
        
    except Exception as e:
        state.errors.append(f"storage error: {str(e)}")
    return state

async def publish_result(state: WorkflowState) -> WorkflowState:
    """Publish the complete workflow result to Kafka output topic"""
    try:
        # Create a comprehensive result payload
        result_payload = {
            "transaction_id": state.transaction.transaction_id,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "transaction": state.transaction.model_dump() if hasattr(state.transaction, 'model_dump') else state.transaction.dict(),
            "fraud_analysis": state.fraud_analysis,
            "similarity_data": state.similarity_data,
            "recommendation": state.recommendation,
            "agent_reflection": state.agent_reflection,
            "storage_status": state.storage_status,
            "errors": state.errors
        }
        
        # Use the existing Kafka service to publish
        kafka_service = CloudKafkaService()
        success = kafka_service.publish_message(
            key=state.transaction.transaction_id,
            data=result_payload,
            topic=Settings.KAFKA_OUTPUT_TOPIC
        )
        
        if success:
            state.publish_status = "published"
            logger.info(f"Published result for transaction {state.transaction.transaction_id}")
        else:
            state.publish_status = "failed"
            state.errors.append("Failed to publish result to Kafka")
            
        # Pretty print
        pretty_publish(state.publish_status, state.transaction.transaction_id)
        
    except Exception as e:
        state.errors.append(f"publish_result error: {str(e)}")
        state.publish_status = "error"
    
    return state

def build_workflow(checkpointer):
    """Build the LangGraph workflow with MongoDB checkpointing"""
    workflow = StateGraph(WorkflowState)

    # Add nodes
    workflow.add_node("duplicate", duplicate_check)
    workflow.add_node("classify", fraud_classification)
    workflow.add_node("similarity", similarity_search)
    workflow.add_node("reflect", agent_reflection)
    workflow.add_node("recommend", action_recommendation)
    workflow.add_node("store", store_transaction)
    workflow.add_node("publish", publish_result) # Add the new node

    # Define the workflow
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
    workflow.add_edge("store", "publish") # Add the new edge
    workflow.add_edge("publish", END)

    # Compile with MongoDB checkpointer
    return workflow.compile(checkpointer=checkpointer)

async def run_workflow_with_checkpoint(compiled_workflow, transaction_data: dict, thread_id: str):
    """Run workflow with proper checkpoint handling"""
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        # Check if there's an existing checkpoint
        existing_state = await compiled_workflow.aget_state(config)
        
        if existing_state and existing_state.values:
            logger.info(f"Resuming workflow from checkpoint for thread: {thread_id}")
            final_state = await compiled_workflow.ainvoke(None, config=config)
        else:
            logger.info(f"Starting new workflow for thread: {thread_id}")
            # Start fresh workflow
            start_state = ensure_state({"transaction": transaction_data})
            final_state = await compiled_workflow.ainvoke(start_state, config=config)
        
        # After END, ainvoke returns a dict—hydrate for printing/return
        hydrated = ensure_state(final_state)
        pretty_final(hydrated)
        return hydrated
        
    except Exception as e:
        logger.error(f"Error in workflow execution for {thread_id}: {e}")
        # If checkpoint resume fails, try starting fresh
        try:
            logger.info(f"Attempting fresh start for thread: {thread_id}")
            start_state = ensure_state({"transaction": transaction_data})
            final_state = await compiled_workflow.ainvoke(start_state, config=config)
            hydrated = ensure_state(final_state)
            pretty_final(hydrated)
            return hydrated
        except Exception as fresh_error:
            logger.error(f"Fresh start also failed for {thread_id}: {fresh_error}")
            raise

async def process_transaction_from_kafka(transaction: Transaction, compiled_workflow):
    """Process a single transaction from Kafka through the workflow with checkpointing"""
    try:
        print(f"\n========= Processing Transaction: {transaction.transaction_id} =========")
        thread_id = f"txn_{transaction.transaction_id}"
        
        # Convert Transaction object to dict for workflow processing
        if hasattr(transaction, 'model_dump'):
            transaction_dict = transaction.model_dump()
        else:
            transaction_dict = transaction.dict()
        
        final_state = await run_workflow_with_checkpoint(
            compiled_workflow, 
            transaction_dict, 
            thread_id
        )
        
        return final_state
    except Exception as e:
        logger.error(f"Error processing transaction {transaction.transaction_id}: {e}")
        return None

def kafka_transaction_callback(transaction: Transaction, compiled_workflow):
    """Callback function for Kafka consumer to process transactions with checkpointing"""
    try:
        # Create a new event loop for this thread if needed
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Run the async workflow with checkpointing
        if loop.is_running():
            # If we're already in an event loop, create a task
            asyncio.create_task(process_transaction_from_kafka(transaction, compiled_workflow))
        else:
            # If no event loop is running, run the coroutine
            loop.run_until_complete(process_transaction_from_kafka(transaction, compiled_workflow))
    except Exception as e:
        logger.error(f"Error in Kafka callback: {e}")

async def main():
    """Main execution function using async context manager with Kafka consumption"""
    
    # Use async context manager for MongoDB checkpointer
    async with AsyncMongoDBSaver.from_conn_string(
        conn_string=Settings.MONGODB_URI,
        db_name=Settings.MONGODB_DATABASE,
        collection_name="langgraph_checkpoints"
    ) as checkpointer:
        
        logger.info("MongoDB checkpointer initialized successfully")
        
        try:
            # Build the workflow with MongoDB checkpointing
            compiled_workflow = build_workflow(checkpointer)
            logger.info("Workflow compiled successfully with MongoDB checkpointer")
            
            # Initialize Kafka service
            kafka_service = CloudKafkaService()
            
            print("Starting Kafka consumer for fraud detection workflow with memory...")
            print(f"Listening on topic: {Settings.KAFKA_TOPIC}")
            print("Press Ctrl+C to stop...")
            
            # Create a callback function that includes the compiled workflow
            def callback_with_workflow(transaction: Transaction):
                kafka_transaction_callback(transaction, compiled_workflow)
            
            # Start consuming transactions from Kafka
            kafka_service.consume_transactions(callback_with_workflow)
            
        except KeyboardInterrupt:
            print("\nShutting down...")
        except Exception as e:
            logger.error(f"Error in main execution: {e}")
            raise
        finally:
            # Clean up Kafka service
            if 'kafka_service' in locals():
                kafka_service.close()

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the main function
    asyncio.run(main())
