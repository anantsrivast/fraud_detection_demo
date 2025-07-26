from langchain.agents import tool
from langgraph.graph import StateGraph, END
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from pymongo import MongoClient
from transaction import Transaction
from fraud_signature_service import FraudSignatureService
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import json
import logging
import asyncio
from settings import settings

logger = logging.getLogger(__name__)

# -------------------- State Definition --------------------

class WorkflowState(BaseModel):
    transaction: Dict[str, Any]
    duplicate_check: Optional[Dict[str, Any]] = None
    fraud_analysis: Optional[Dict[str, Any]] = None
    similarity_data: Optional[Dict[str, Any]] = None
    recommendation: Optional[str] = None
    storage_status: Optional[str] = None
    errors: List[str] = []

# -------------------- Tool Implementations --------------------

def get_db():
    client = MongoClient(settings.MONGODB_URI)
    return client[settings.MONGODB_DATABASE]

@tool
async def duplicate_check(state: WorkflowState) -> WorkflowState:
    try:
        transaction_data = state.transaction
        if not transaction_data:
            raise ValueError("No transaction data provided")

        transaction = Transaction(**transaction_data)

        def check_duplicate():
            db = get_db()
            return db[settings.MONGODB_COLLECTION_CASES].count_documents({"transaction_id": transaction.transaction_id}) > 0

        loop = asyncio.get_event_loop()
        is_duplicate = await loop.run_in_executor(None, check_duplicate)

        result = {
            "is_duplicate": is_duplicate,
            "customer_id": transaction.customer_id,
            "transaction_id": transaction.transaction_id,
            "checked_at": loop.time()
        }

        if is_duplicate:
            result["recommendation"] = "SKIP_PROCESSING"
            result["reason"] = "Duplicate complaint detected within 24 hours"
        else:
            result["recommendation"] = "CONTINUE_PROCESSING"

        logger.info(f"Duplicate check completed for {transaction.transaction_id}: {is_duplicate}")
        state.duplicate_check = result
    except Exception as e:
        state.errors.append(f"duplicate_check error: {str(e)}")
    return state

@tool
async def fraud_classification(state: WorkflowState) -> WorkflowState:
    try:
        amt = state.transaction.get("amount", 0)
        state.fraud_analysis = {"is_fraud": amt > 1000}
    except Exception as e:
        state.errors.append(f"fraud_classification error: {str(e)}")
    return state

@tool
async def similarity_search(state: WorkflowState) -> WorkflowState:
    try:
        loop = asyncio.get_event_loop()
        encoder = SentenceTransformer(settings.EMBEDDING_MODEL)
        fs = FraudSignatureService()
        signatures = fs.generate_fraud_signatures(Transaction(**state.transaction))

        def encode():
            return encoder.encode([" ".join(signatures)])[0]

        vector = await loop.run_in_executor(None, encode)

        def search():
            db = get_db()
            return list(db[settings.MONGODB_COLLECTION_SIGNATURES].aggregate([
                {"$vectorSearch": {
                    "index": settings.VECTOR_INDEX_NAME,
                    "path": "vector",
                    "queryVector": vector,
                    "numCandidates": 50,
                    "limit": 5,
                }}
            ]))

        results = await loop.run_in_executor(None, search)
        state.similarity_data = {"similar_cases": results}

        def insert_signature():
            db = get_db()
            db[settings.MONGODB_COLLECTION_SIGNATURES].insert_one({
                "transaction_id": state.transaction["transaction_id"],
                "signatures": signatures,
                "vector": vector
            })

        if state.fraud_analysis and state.fraud_analysis.get("is_fraud"):
            await loop.run_in_executor(None, insert_signature)

    except Exception as e:
        state.errors.append(f"similarity_search error: {str(e)}")
    return state

@tool
async def action_recommendation(state: WorkflowState) -> WorkflowState:
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0)
        context = f"Fraud: {state.fraud_analysis}\nSimilar: {state.similarity_data}"
        prompt = f"Given the analysis: {context}, what should the fraud response team do?"
        state.recommendation = await llm.ainvoke(prompt)
    except Exception as e:
        state.errors.append(f"action_recommendation error: {str(e)}")
    return state

@tool
async def store_transaction(state: WorkflowState) -> WorkflowState:
    try:
        def store():
            db = get_db()
            db[settings.MONGODB_COLLECTION_CASES].insert_one(state.transaction)

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, store)
        state.storage_status = "stored"
    except Exception as e:
        state.errors.append(f"storage error: {str(e)}")
    return state

# -------------------- LangGraph Definition --------------------

def build_workflow():
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
    compiled = build_workflow()
    with open("merged_transactions_array.json") as f:
        txns = json.load(f)

    async def run():
        for txn in txns[:5]:
            print("\n=========")
            state = WorkflowState(transaction=txn)
            final_state = await compiled.ainvoke(state)
            print("Final:", final_state)

    asyncio.run(run())
