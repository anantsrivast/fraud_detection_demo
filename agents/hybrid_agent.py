from langchain.agents import tool
from langgraph.graph import StateGraph, END
from langchain_community.chat_models import ChatOpenAI  # Updated import
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

    @model_validator(mode="before")
    @classmethod
    def convert_transaction_if_needed(cls, values):
        txn = values.get("transaction")
        if isinstance(txn, dict):
            values["transaction"] = Transaction(**txn)
        return values

# ---------------- Mongo ----------------

def get_db():
    """
    Establishes and returns a MongoDB database connection.
    """
    client = MongoClient(Settings.MONGODB_URI)
    return client[Settings.MONGODB_DATABASE]

# ---------------- Tool Definitions ----------------

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


async def fraud_classification(state: WorkflowState) -> WorkflowState:
    """
    Applies a simple rule-based fraud check based on transaction amount.
    Flags transaction as fraudulent if amount > 1000.
    """
    try:
        amt = state.transaction.amount or 0
        state.fraud_analysis = {"is_fraud": amt > 1000}
    except Exception as e:
        state.errors.append(f"fraud_classification error: {str(e)}")
    return state


async def similarity_search(state: WorkflowState) -> WorkflowState:
    """
    Encodes transaction fraud signatures and searches for similar cases
    using MongoDB vector search. Optionally stores the embedding.
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


async def action_recommendation(state: WorkflowState) -> WorkflowState:
    """
    Invokes an LLM to generate an action recommendation for the fraud response team
    based on previous analysis and similarity search results.
    """
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        context = f"Fraud: {state.fraud_analysis}\nSimilar: {state.similarity_data}"
        prompt = f"Given the analysis: {context}, what should the fraud response team do?"
        state.recommendation = await llm.ainvoke(prompt)
    except Exception as e:
        state.errors.append(f"action_recommendation error: {str(e)}")
    return state


async def store_transaction(state: WorkflowState) -> WorkflowState:
    """
    Persists the transaction to MongoDB for audit or review purposes.
    """
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

# -------------------- LangGraph Definition --------------------

def build_workflow():
    """
    Constructs and compiles the LangGraph workflow with logic for fraud analysis.

    Returns:
        A compiled graph ready to invoke with WorkflowState.
    """
    workflow = StateGraph(WorkflowState)
    workflow.add_node("duplicate", duplicate_check)
    workflow.add_node("classify", fraud_classification)
    workflow.add_node("similarity", similarity_search)
    workflow.add_node("recommend", action_recommendation)
    workflow.add_node("store", store_transaction)

    workflow.set_entry_point("duplicate")
    workflow.add_conditional_edges(
        "duplicate",
        lambda state: "skip" if state.duplicate_check and state.duplicate_check.get("is_duplicate") else "continue",
        {"skip": END, "continue": "classify"}
    )
    workflow.add_conditional_edges(
        "classify",
        lambda state: "continue" if state.fraud_analysis and state.fraud_analysis.get("is_fraud") else "end",
        {"continue": "similarity", "end": END}
    )
    workflow.add_edge("similarity", "recommend")
    workflow.add_edge("recommend", "store")
    workflow.add_edge("store", END)

    return workflow.compile()

# -------------------- Execution --------------------

if __name__ == "__main__":
    """
    Executes the compiled workflow on the first 5 transactions in the input JSON file.
    """
    compiled = build_workflow()
    with open("merged_transactions.json") as f:
        txns = json.load(f)

    async def run():
        for txn in txns[:5]:
            print("\n=========")
            state = WorkflowState(transaction=txn)
            final_state = await compiled.ainvoke(state)
            print("Final:", final_state)

    asyncio.run(run())
