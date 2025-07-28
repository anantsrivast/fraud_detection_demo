from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # Use async version
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from models.transaction import Transaction
from services.fraud_signature_service import FraudSignatureService
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, model_validator
import json
import logging
import asyncio
from config.settings import Settings

logger = logging.getLogger(__name__)

class WorkflowState(BaseModel):
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
            "checked_at": loop.time(),
            "recommendation": "SKIP_PROCESSING" if is_duplicate else "CONTINUE_PROCESSING"
        }

        if is_duplicate:
            result["reason"] = "Duplicate complaint detected within 24 hours"

        state.duplicate_check = result
        logger.info(f"Duplicate check completed for {transaction.transaction_id}: {is_duplicate}")
        
        print(f"\n=== DUPLICATE CHECK COMPLETE ===")
        print(f"State after duplicate_check: {state}")

    except Exception as e:
        state.errors.append(f"duplicate_check error: {str(e)}")

    return state

async def fraud_classification(state: WorkflowState) -> WorkflowState:
    try:
        amt = state.transaction.amount or 0
        state.fraud_analysis = {"is_fraud": amt > 1000}
        logger.info(f"Fraud classification: {state.fraud_analysis}")
        
        print(f"\n=== FRAUD CLASSIFICATION COMPLETE ===")
        print(f"State after fraud_classification: {state}")
        
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
            # Convert ObjectId to string to avoid serialization issues
            results = list(db[Settings.MONGODB_COLLECTION_SIGNATURES].aggregate([
                {"$vectorSearch": {
                    "index": Settings.VECTOR_INDEX_NAME,
                    "path": "embedding",
                    "queryVector": embedding.tolist(),  # Convert numpy array to list
                    "numCandidates": 50,
                    "limit": 5,
                }}
            ]))
            
            # Convert ObjectId to string for serialization
            for result in results:
                if '_id' in result:
                    result['_id'] = str(result['_id'])
            
            return results

        results = await loop.run_in_executor(None, search)
        state.similarity_data = {"similar_cases": results}

        def insert_signature():
            db = get_db()
            db[Settings.MONGODB_COLLECTION_SIGNATURES].insert_one({
                "transaction_id": state.transaction.transaction_id,
                "signatures": signatures,
                "embedding": embedding.tolist()  # Convert numpy array to list
            })

        if state.fraud_analysis and state.fraud_analysis.get("is_fraud"):
            await loop.run_in_executor(None, insert_signature)
            logger.info(f"Stored fraud signature for {state.transaction.transaction_id}")

        print(f"\n=== SIMILARITY SEARCH COMPLETE ===")
        print(f"State after similarity_search: {state}")

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
        
        print(f"\n=== AGENT REFLECTION COMPLETE ===")
        print(f"State after agent_reflection: {state}")

    except Exception as e:
        logger.warning(f"agent_reflection error (continuing): {str(e)}")
        # Provide a simple fallback response
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        state.agent_reflection = f"Automatic analysis: {'High risk transaction - requires immediate review' if fraud_status else 'Low risk transaction - standard processing'}"
        state.errors.append(f"agent_reflection error: {str(e)}")
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
        
        print(f"\n=== ACTION RECOMMENDATION COMPLETE ===")
        print(f"State after action_recommendation: {state}")

    except Exception as e:
        logger.warning(f"action_recommendation error (continuing): {str(e)}")
        # Provide a simple fallback response
        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        if fraud_status:
            state.recommendation = "RECOMMENDED ACTION: Block transaction, investigate customer account, review for fraud patterns."
        else:
            state.recommendation = "RECOMMENDED ACTION: Approve transaction, continue monitoring."
        state.errors.append(f"action_recommendation error: {str(e)}")
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
        
        print(f"\n=== STORE TRANSACTION COMPLETE ===")
        print(f"State after store_transaction: {state}")
        
    except Exception as e:
        state.errors.append(f"storage error: {str(e)}")
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
    workflow.add_edge("store", END)

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
            # Resume from existing checkpoint
            final_state = await compiled_workflow.ainvoke(None, config=config)
        else:
            logger.info(f"Starting new workflow for thread: {thread_id}")
            # Start fresh workflow
            initial_state = WorkflowState(transaction=transaction_data)
            final_state = await compiled_workflow.ainvoke(initial_state, config=config)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Error in workflow execution for {thread_id}: {e}")
        # If checkpoint resume fails, try starting fresh
        try:
            logger.info(f"Attempting fresh start for thread: {thread_id}")
            initial_state = WorkflowState(transaction=transaction_data)
            final_state = await compiled_workflow.ainvoke(initial_state, config=config)
            return final_state
        except Exception as fresh_error:
            logger.error(f"Fresh start also failed for {thread_id}: {fresh_error}")
            raise

async def main():
    """Main execution function using async context manager"""
    
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
            
            # Load transaction data
            with open("merged_transactions.json") as f:
                txns = json.load(f)

            # Process transactions with better rate limiting
            for i, txn in enumerate(txns[:5]):
                print(f"\n{'='*50}")
                thread_id = f"txn_{txn['transaction_id']}"
                
                try:
                    logger.info(f"Processing transaction {i+1}/5: {txn['transaction_id']}")
                    
                    # Add longer delay between transactions to respect rate limits
                    if i > 0:
                        print("Waiting to respect rate limits...")
                        await asyncio.sleep(5)  # Increased delay
                    
                    final_state = await run_workflow_with_checkpoint(
                        compiled_workflow, 
                        txn, 
                        thread_id
                    )
                    
                    print(f"\n=== FINAL STATE ===")
                    print(f"Final state for {txn['transaction_id']}: {final_state}")
                    print(f"====================\n")
                        
                except Exception as e:
                    logger.error(f"Failed to process transaction {txn.get('transaction_id', 'unknown')}: {e}")
                    print(f"Error processing transaction: {e}")
                    # Continue with next transaction instead of stopping
                    
        except Exception as e:
            logger.error(f"Main execution failed: {e}")
            raise

if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run the main function
    asyncio.run(main())