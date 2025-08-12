from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver  # Use async version
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from models.transaction import Transaction
from services.fraud_signature_service import FraudSignatureService
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, model_validator
import json
import logging
import asyncio
from config.settings import Settings
from langchain_openai import ChatOpenAI
from config.logger import get_logger
from datetime import datetime
import time

logger = get_logger(__name__)
client = MongoClient(Settings.MONGODB_URI)
encoder = SentenceTransformer(Settings.EMBEDDING_MODEL)
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

def pretty_final(state: "WorkflowState") -> None:  # type: ignore
    hr("Final Summary")
    kv("Transaction ID", state.transaction.transaction_id)
    kv("Duplicate", yes_no(state.duplicate_check.get("is_duplicate") if state.duplicate_check else False))
    kv("Fraud", yes_no(state.fraud_analysis.get("is_fraud") if state.fraud_analysis else False))
    kv("Similar Cases", len((state.similarity_data or {}).get("similar_cases", [])))
    kv("Recommendation", state.recommendation or "N/A")
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
    errors: List[str] = Field(default_factory=list)  # avoid shared mutable default
    agent_reflection: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def convert_transaction_if_needed(cls, values):
        txn = values.get("transaction")
        if isinstance(txn, dict):
            values["transaction"] = Transaction(**txn)
        return values


# ---------------------------
# DB
# ---------------------------
def get_db():
    return client[Settings.MONGODB_DATABASE]

# ---------------------------
# Nodes
# ---------------------------

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
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_retries=3,
            request_timeout=60,
            max_tokens=200
        )

        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        similar_count = len(state.similarity_data.get("similar_cases", [])) if state.similarity_data else 0

        prompt = f"""Fraud detected: {fraud_status}
Similar cases found: {similar_count}

Brief analysis (max 100 words): What actions should be taken?"""

        await asyncio.sleep(2)

        response = await llm.ainvoke(prompt)
        state.agent_reflection = str(response.content if hasattr(response, "content") else response)
        logger.info("Agent reflection completed")

        # Pretty print
        pretty_reflection(state.agent_reflection)

    except Exception as e:
        logger.warning(f"agent_reflection error (continuing): {str(e)}")
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
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            max_retries=3,
            request_timeout=60,
            max_tokens=150
        )

        fraud_status = state.fraud_analysis.get("is_fraud", False) if state.fraud_analysis else False
        similar_count = len(state.similarity_data.get("similar_cases", [])) if state.similarity_data else 0

        prompt = f"""Transaction risk: {'HIGH' if fraud_status else 'LOW'}
Similar cases: {similar_count}

Recommended action (max 50 words):"""

        await asyncio.sleep(2)

        response = await llm.ainvoke(prompt)
        state.recommendation = str(response.content if hasattr(response, "content") else response)
        logger.info("Action recommendation completed")

        # Pretty print
        pretty_recommendation(state.recommendation)

    except Exception as e:
        logger.warning(f"action_recommendation error (continuing): {str(e)}")
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


# ---------------------------
# Graph
# ---------------------------

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

    return workflow.compile(checkpointer=checkpointer)


# ---------------------------
# Runner
# ---------------------------

async def run_workflow_with_checkpoint(compiled_workflow, transaction_data: dict, thread_id: str):
    """Run workflow with proper checkpoint handling"""

    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    try:
        existing_state = await compiled_workflow.aget_state(config)

        if existing_state and existing_state.values:
            logger.info(f"Resuming workflow from checkpoint for thread: {thread_id}")
            final_state = await compiled_workflow.ainvoke(None, config=config)
        else:
            logger.info(f"Starting new workflow for thread: {thread_id}")
            start_state = ensure_state({"transaction": transaction_data})
            final_state = await compiled_workflow.ainvoke(start_state, config=config)

        # After END, ainvoke returns a dict—hydrate for printing/return
        hydrated = ensure_state(final_state)
        pretty_final(hydrated)
        return hydrated

    except Exception as e:
        logger.error(f"Error in workflow execution for {thread_id}: {e}")
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


async def main():
    """Main execution function using async context manager"""

    async with AsyncMongoDBSaver.from_conn_string(
        conn_string=Settings.MONGODB_URI,
        db_name=Settings.MONGODB_DATABASE,
        collection_name="langgraph_checkpoints"
    ) as checkpointer:

        logger.info("MongoDB checkpointer initialized successfully")

        try:
            compiled_workflow = build_workflow(checkpointer)
            logger.info("Workflow compiled successfully with MongoDB checkpointer")

            with open("merged_transactions.json") as f:
                txns = json.load(f)

            for i, txn in enumerate(txns[:5]):
                print(f"\n{'='*50}")
                thread_id = f"txn_{txn['transaction_id']}"

                try:
                    logger.info(f"Processing transaction {i+1}/5: {txn['transaction_id']}")

                    if i > 0:
                        print("Waiting to respect rate limits...")
                        await asyncio.sleep(5)

                    final_state = await run_workflow_with_checkpoint(
                        compiled_workflow,
                        txn,
                        thread_id
                    )
                    # Already pretty-printed inside run_workflow_with_checkpoint

                except Exception as e:
                    logger.error(f"Failed to process transaction {txn.get('transaction_id', 'unknown')}: {e}")
                    print(f"Error processing transaction: {e}")

        except Exception as e:
            logger.error(f"Main execution failed: {e}")
            raise


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
