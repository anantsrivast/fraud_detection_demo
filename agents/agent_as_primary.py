from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langgraph_checkpoint_mongodb.aio import AsyncMongoDBSaver
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.tools import tool
from sentence_transformers import SentenceTransformer
from motor.motor_asyncio import AsyncIOMotorClient
from models.transaction import Transaction
from services.fraud_signature_service import FraudSignatureService
from typing import Dict, Any, List, Optional, Annotated
from pydantic import BaseModel, Field
import json
import logging
import asyncio
from config.settings import Settings
from config.logger import get_logger
import time
from datetime import datetime

logger = get_logger(__name__)

# ========================================
# SIMPLIFIED STATE FOR DETERMINISTIC LLM WORKFLOW
# ========================================

class DeterministicLLMState(BaseModel):
    # Core transaction data
    transaction: Transaction
    
    # LLM conversation messages
    messages: Annotated[List[Any], Field(default_factory=list)]
    
    # Analysis results (populated by tools)
    duplicate_check: Optional[Dict[str, Any]] = None
    fraud_analysis: Optional[Dict[str, Any]] = None
    similarity_data: Optional[Dict[str, Any]] = None
    storage_status: Optional[str] = None
    
    # Final outputs
    agent_reflection: Optional[str] = None
    recommendation: Optional[str] = None
    
    # Control flow
    next_action: str = "start"  # start, tools_needed, reflect, complete
    errors: List[str] = Field(default_factory=list)

# ========================================
# ASYNC DATABASE CONNECTION
# ========================================

async_mongo_client = None

async def get_async_db():
    global async_mongo_client
    if async_mongo_client is None:
        async_mongo_client = AsyncIOMotorClient(Settings.MONGODB_URI)
    return async_mongo_client[Settings.MONGODB_DATABASE]

# ========================================
# TOOLS - Same as before but simplified
# ========================================

@tool
async def check_duplicate_transaction(ip_address: str, customer_id: str) -> Dict[str, Any]:
    """
    Check if a transaction is a duplicate based on IP address and customer ID.
    
    Args:
        ip_address: The IP address of the transaction
        customer_id: The customer ID for the transaction
    """
    try:
        db = await get_async_db()
        count = await db[Settings.MONGODB_COLLECTION_CASES].count_documents({
            "ip_address": ip_address,
            "customer_id": customer_id
        })
        
        is_duplicate = count > 0
        
        return {
            "is_duplicate": is_duplicate,
            "customer_id": customer_id,
            "ip_address": ip_address,
            "duplicate_count": count,
            "checked_at": time.time(),
            "recommendation": "SKIP_PROCESSING" if is_duplicate else "CONTINUE_PROCESSING",
            "reason": f"Found {count} duplicate transactions" if is_duplicate else "No duplicates found"
        }
        
    except Exception as e:
        logger.error(f"Error in duplicate check: {e}")
        return {"error": str(e), "is_duplicate": False, "recommendation": "CONTINUE_PROCESSING"}

@tool
async def analyze_fraud_risk(transaction_id: str, amount: float, customer_id: str) -> Dict[str, Any]:
    """
    Analyze a transaction for fraud risk based on amount and customer history.
    
    Args:
        transaction_id: Unique transaction identifier
        amount: Transaction amount
        customer_id: Customer identifier
    """
    try:
        risk_factors = []
        risk_score = 0.0
        
        # Amount-based risk (your existing logic)
        if amount > 1000:
            risk_factors.append("high_amount")
            risk_score += 0.6
            
        if amount > 5000:
            risk_factors.append("very_high_amount") 
            risk_score += 0.3
            
        is_fraud = risk_score >= 0.5  # Your existing threshold
        
        return {
            "transaction_id": transaction_id,
            "is_fraud": is_fraud,
            "risk_score": min(risk_score, 1.0),
            "risk_factors": risk_factors,
            "confidence": 0.8 if risk_score > 0.7 else 0.6
        }
        
    except Exception as e:
        logger.error(f"Error in fraud analysis: {e}")
        return {"error": str(e), "is_fraud": False, "risk_score": 0.0}

@tool
async def search_similar_fraud_cases(transaction_id: str, amount: float, ip_address: str, customer_id: str) -> Dict[str, Any]:
    """
    Search for similar fraud cases using vector similarity.
    
    Args:
        transaction_id: Unique transaction identifier
        amount: Transaction amount
        ip_address: IP address
        customer_id: Customer ID
    """
    try:
        # Create temporary transaction for signature generation
        temp_transaction = Transaction(
            transaction_id=transaction_id,
            amount=amount,
            ip_address=ip_address,
            customer_id=customer_id
        )
        
        # Generate signatures and embeddings
        encoder = SentenceTransformer(Settings.EMBEDDING_MODEL)
        fs = FraudSignatureService()
        signatures = fs.generate_fraud_signatures(temp_transaction)
        
        # Run embedding in executor
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None,
            lambda: encoder.encode([" ".join(signatures)])[0]
        )
        
        # Search database
        db = await get_async_db()
        cursor = db[Settings.MONGODB_COLLECTION_SIGNATURES].aggregate([
            {"$vectorSearch": {
                "index": Settings.VECTOR_INDEX_NAME,
                "path": "embedding",
                "queryVector": embedding.tolist(),
                "numCandidates": 50,
                "limit": 5,
            }}
        ])
        
        similar_cases = []
        async for result in cursor:
            if '_id' in result:
                result['_id'] = str(result['_id'])
            similar_cases.append(result)
        
        # Store signature for future searches
        await db[Settings.MONGODB_COLLECTION_SIGNATURES].insert_one({
            "transaction_id": transaction_id,
            "signatures": signatures,
            "embedding": embedding.tolist(),
            "created_at": datetime.utcnow()
        })
        
        return {
            "transaction_id": transaction_id,
            "similar_cases": similar_cases,
            "similar_count": len(similar_cases),
            "signatures_used": signatures
        }
        
    except Exception as e:
        logger.error(f"Error in similarity search: {e}")
        return {"error": str(e), "similar_cases": [], "similar_count": 0}

@tool
async def store_transaction_data(transaction_id: str, customer_id: str, amount: float, ip_address: str) -> Dict[str, Any]:
    """
    Store transaction data in the database.
    
    Args:
        transaction_id: Unique transaction identifier
        customer_id: Customer ID
        amount: Transaction amount
        ip_address: IP address
    """
    try:
        db = await get_async_db()
        
        # Store transaction
        transaction_doc = {
            "transaction_id": transaction_id,
            "customer_id": customer_id,
            "amount": amount,
            "ip_address": ip_address,
            "processed_at": datetime.utcnow(),
            "processing_method": "llm_deterministic"
        }
        
        result = await db[Settings.MONGODB_COLLECTION_CASES].insert_one(transaction_doc)
        
        return {
            "stored": True,
            "document_id": str(result.inserted_id),
            "stored_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error storing transaction: {e}")
        return {"stored": False, "error": str(e)}

# ========================================
# LLM NODE - Contains all the business logic in the prompt
# ========================================

async def llm_decision_node(state: DeterministicLLMState) -> DeterministicLLMState:
    """
    LLM node that follows deterministic business rules specified in the prompt
    """
    try:
        # Create tools list
        tools = [
            check_duplicate_transaction,
            analyze_fraud_risk,
            search_similar_fraud_cases,
            store_transaction_data
        ]
        
        # Create LLM with tools
        llm = ChatOpenAI(
            model="gpt-4",
            temperature=0,  # Deterministic responses
            max_retries=3,
            request_timeout=120
        ).bind_tools(tools)
        
        # Build deterministic system prompt with exact business rules
        system_prompt = f"""You are a fraud detection system that MUST follow these exact rules:

DETERMINISTIC WORKFLOW RULES (FOLLOW EXACTLY):

1. DUPLICATE CHECK (ALWAYS FIRST):
   - Call check_duplicate_transaction with ip_address and customer_id
   - IF is_duplicate = true: STOP here, set next_action="complete", provide recommendation="SKIP_PROCESSING"
   - IF is_duplicate = false: Continue to step 2

2. FRAUD CLASSIFICATION (IF NOT DUPLICATE):
   - Call analyze_fraud_risk with transaction_id, amount, customer_id
   - IF is_fraud = false: SKIP steps 3&4, go to step 5 (no storage needed for clean transactions)
   - IF is_fraud = true: Continue to step 3

3. SIMILARITY SEARCH (ONLY IF FRAUD DETECTED):
   - Call search_similar_fraud_cases with transaction details
   - This provides context for final recommendation

4. STORE TRANSACTION (ONLY IF FRAUD DETECTED):
   - Call store_transaction_data to save the fraudulent transaction
   - Only fraudulent transactions need to be stored for investigation

5. FINAL RECOMMENDATION:
   - Based on all results, provide final recommendation and reflection
   - Set next_action="complete"

CURRENT TRANSACTION:
- ID: {state.transaction.transaction_id}
- Customer: {state.transaction.customer_id} 
- Amount: ${state.transaction.amount}
- IP: {state.transaction.ip_address}

IMPORTANT: 
- Follow the rules EXACTLY in order
- Don't skip steps unless rules specify
- Be deterministic - same input should give same output
- Only call tools, don't make assumptions about results
"""

        # Initialize conversation if needed
        if not state.messages:
            state.messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=f"Analyze transaction {state.transaction.transaction_id} following the deterministic workflow rules.")
            ]
        
        # Get LLM response
        response = await llm.ainvoke(state.messages)
        state.messages.append(response)
        
        # Determine next action based on response
        if response.tool_calls:
            state.next_action = "tools_needed"
            logger.info(f"LLM requesting {len(response.tool_calls)} tool calls")
        else:
            # LLM provided final analysis without tools
            state.next_action = "complete"
            
            # Extract recommendation from response content
            response_text = response.content if hasattr(response, 'content') else str(response)
            state.agent_reflection = response_text
            
            # Parse recommendation (simple keyword extraction)
            if "SKIP_PROCESSING" in response_text.upper():
                state.recommendation = "SKIP_PROCESSING"
            elif "APPROVE" in response_text.upper():
                state.recommendation = "APPROVE"
            elif "REJECT" in response_text.upper():
                state.recommendation = "REJECT"
            elif "REVIEW" in response_text.upper():
                state.recommendation = "MANUAL_REVIEW"
            else:
                state.recommendation = "NEEDS_ANALYSIS"
            
            logger.info(f"LLM final decision: {state.recommendation}")
        
        print(f"\n=== LLM DECISION NODE ===")
        print(f"Next action: {state.next_action}")
        print(f"Tool calls requested: {len(response.tool_calls) if response.tool_calls else 0}")
        
    except Exception as e:
        error_msg = f"Error in LLM decision node: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.next_action = "complete"
        state.recommendation = "ERROR"
    
    return state

# ========================================
# TOOLS NODE - Executes all tools and updates state
# ========================================

async def execute_tools_node(state: DeterministicLLMState) -> DeterministicLLMState:
    """
    Execute all requested tools and update state with results
    """
    try:
        # Get the last AI message with tool calls
        last_message = state.messages[-1]
        
        if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
            logger.warning("No tool calls found in last message")
            state.next_action = "complete"
            return state
        
        # Create tool node
        tools = [
            check_duplicate_transaction,
            analyze_fraud_risk,
            search_similar_fraud_cases,
            store_transaction_data
        ]
        tool_node = ToolNode(tools)
        
        # Execute tools
        tool_results = await tool_node.ainvoke({"messages": state.messages})
        
        # Add tool results to conversation
        state.messages.extend(tool_results["messages"])
        
        # Update state based on tool results
        for tool_call, tool_result in zip(last_message.tool_calls, tool_results["messages"]):
            if isinstance(tool_result, ToolMessage):
                tool_name = tool_call["name"]
                result_data = json.loads(tool_result.content)
                
                # Update state based on which tool was called
                if tool_name == "check_duplicate_transaction":
                    state.duplicate_check = result_data
                    logger.info(f"Duplicate check: {result_data.get('is_duplicate', 'unknown')}")
                    
                elif tool_name == "analyze_fraud_risk":
                    state.fraud_analysis = result_data
                    logger.info(f"Fraud analysis: {result_data.get('is_fraud', 'unknown')}")
                    
                elif tool_name == "search_similar_fraud_cases":
                    state.similarity_data = result_data
                    logger.info(f"Similar cases found: {result_data.get('similar_count', 0)}")
                    
                elif tool_name == "store_transaction_data":
                    state.storage_status = "stored" if result_data.get("stored") else "failed"
                    logger.info(f"Storage status: {state.storage_status}")
        
        # Determine next action - send back to LLM to continue workflow
        state.next_action = "continue_llm"
        
        print(f"\n=== TOOLS EXECUTED ===")
        print(f"Duplicate check: {state.duplicate_check}")
        print(f"Fraud analysis: {state.fraud_analysis}")
        print(f"Similarity data: {state.similarity_data}")
        print(f"Storage status: {state.storage_status}")
        
    except Exception as e:
        error_msg = f"Error executing tools: {str(e)}"
        logger.error(error_msg)
        state.errors.append(error_msg)
        state.next_action = "complete"
    
    return state

# ========================================
# ROUTING LOGIC
# ========================================

def route_next_step(state: DeterministicLLMState) -> str:
    """Simple routing based on next_action"""
    if state.next_action == "tools_needed":
        return "tools"
    elif state.next_action == "continue_llm":
        return "llm"
    else:  # complete, error, etc.
        return "end"

# ========================================
# BUILD DETERMINISTIC LLM WORKFLOW
# ========================================

def build_deterministic_llm_workflow(checkpointer):
    """Build the 3-node deterministic LLM workflow: LLM → Tools → End"""
    
    workflow = StateGraph(DeterministicLLMState)
    
    # Add the 3 nodes
    workflow.add_node("llm", llm_decision_node)
    workflow.add_node("tools", execute_tools_node)
    # END is implicit
    
    # Set entry point
    workflow.set_entry_point("llm")
    
    # Add routing
    workflow.add_conditional_edges(
        "llm",
        route_next_step,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "tools", 
        route_next_step,
        {
            "llm": "llm",
            "end": END
        }
    )
    
    return workflow.compile(checkpointer=checkpointer)

# ========================================
# MAIN EXECUTION
# ========================================

async def run_deterministic_llm_workflow(compiled_workflow, transaction_data: dict, thread_id: str):
    """Run the deterministic LLM workflow"""
    
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }
    
    try:
        logger.info(f"Starting deterministic LLM workflow for thread: {thread_id}")
        initial_state = DeterministicLLMState(transaction=transaction_data)
        final_state = await compiled_workflow.ainvoke(initial_state, config=config)
        
        return final_state
        
    except Exception as e:
        logger.error(f"Error in deterministic LLM workflow for {thread_id}: {e}")
        raise

async def main():
    """Main execution function"""
    
    async with AsyncMongoDBSaver.from_conn_string(
        conn_string=Settings.MONGODB_URI,
        db_name=Settings.MONGODB_DATABASE,
        collection_name="deterministic_llm_checkpoints"
    ) as checkpointer:
        
        logger.info("Deterministic LLM workflow initialized")
        
        try:
            # Build workflow
            compiled_workflow = build_deterministic_llm_workflow(checkpointer)
            logger.info("Deterministic LLM workflow compiled")
            
            # Load transaction data
            with open("merged_transactions.json") as f:
                txns = json.load(f)
            
            # Process transactions
            for i, txn in enumerate(txns[:3]):
                print(f"\n{'='*60}")
                print(f"DETERMINISTIC LLM PROCESSING: {txn['transaction_id']}")
                print(f"{'='*60}")
                
                thread_id = f"det_llm_{txn['transaction_id']}"
                
                try:
                    if i > 0:
                        await asyncio.sleep(5)  # Rate limiting
                    
                    final_state = await run_deterministic_llm_workflow(
                        compiled_workflow,
                        txn,
                        thread_id
                    )
                    
                    print(f"\n=== FINAL DETERMINISTIC RESULTS ===")
                    print(f"Transaction: {txn['transaction_id']}")
                    print(f"Duplicate Check: {final_state.duplicate_check}")
                    print(f"Fraud Analysis: {final_state.fraud_analysis}")
                    print(f"Similarity Data: {'Found' if final_state.similarity_data else 'None'}")
                    print(f"Storage Status: {final_state.storage_status}")
                    print(f"Final Recommendation: {final_state.recommendation}")
                    if final_state.errors:
                        print(f"Errors: {final_state.errors}")
                    print(f"==================================\n")
                    
                except Exception as e:
                    logger.error(f"Failed to process transaction {txn.get('transaction_id')}: {e}")
                    
        except Exception as e:
            logger.error(f"Main execution failed: {e}")
            raise
        finally:
            if async_mongo_client:
                async_mongo_client.close()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())