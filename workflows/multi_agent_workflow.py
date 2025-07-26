from langgraph import StateGraph, END
from typing import Dict, Any, List, TypedDict, Optional
import asyncio
import logging
from datetime import datetime

from agents.duplicate_detection_agent import DuplicateDetectionAgent
from agents.fraud_classification_agent import FraudClassificationAgent
from agents.similarity_search_agent import SimilaritySearchAgent
from agents.action_recommendation_agent import ActionRecommendationAgent
from agents.data_storage_agent import DataStorageAgent
from models.transaction import Transaction
from config.settings import settings

logger = logging.getLogger(__name__)

class MultiAgentState(TypedDict):
    transaction: Dict[str, Any]
    duplicate_check: Optional[Dict[str, Any]]
    fraud_analysis: Optional[Dict[str, Any]]
    similarity_data: Optional[Dict[str, Any]]
    recommendation: Optional[Dict[str, Any]]
    storage_results: Optional[Dict[str, Any]]
    workflow_status: str
    errors: List[str]
    execution_log: List[Dict[str, Any]]

class MultiAgentFraudWorkflow:
    def __init__(self, 
                 duplicate_agent: DuplicateDetectionAgent,
                 classification_agent: FraudClassificationAgent,
                 similarity_agent: SimilaritySearchAgent,
                 recommendation_agent: ActionRecommendationAgent,
                 storage_agent: DataStorageAgent):
        
        self.agents = {
            "duplicate": duplicate_agent,
            "classification": classification_agent,
            "similarity": similarity_agent,
            "recommendation": recommendation_agent,
            "storage": storage_agent
        }
        
        self.workflow = self._build_workflow()
        self.active_executions = {}  # Track concurrent executions
    
    def _build_workflow(self) -> StateGraph:
        """Build the multi-agent workflow using LangGraph"""
        workflow = StateGraph(MultiAgentState)
        
        # Add agent nodes
        workflow.add_node("duplicate_detection", self._run_duplicate_detection)
        workflow.add_node("fraud_classification", self._run_fraud_classification)
        workflow.add_node("similarity_search", self._run_similarity_search)
        workflow.add_node("action_recommendation", self._run_action_recommendation)
        workflow.add_node("data_storage", self._run_data_storage)
        
        # Define workflow edges
        workflow.set_entry_point("duplicate_detection")
        
        # Conditional flow based on duplicate detection
        workflow.add_conditional_edges(
            "duplicate_detection",
            self._should_continue_after_duplicate,
            {
                "continue": "fraud_classification",
                "skip": END
            }
        )
        
        # Run classification and similarity search in parallel
        workflow.add_edge("fraud_classification", "similarity_search")
        workflow.add_edge("similarity_search", "action_recommendation")
        workflow.add_edge("action_recommendation", "data_storage")
        workflow.add_edge("data_storage", END)
        
        return workflow.compile()
    
    async def _run_duplicate_detection(self, state: MultiAgentState) -> MultiAgentState:
        """Run duplicate detection agent"""
        return await self._execute_agent("duplicate", state)
    
    async def _run_fraud_classification(self, state: MultiAgentState) -> MultiAgentState:
        """Run fraud classification agent"""
        return await self._execute_agent("classification", state)
    
    async def _run_similarity_search(self, state: MultiAgentState) -> MultiAgentState:
        """Run similarity search agent"""
        return await self._execute_agent("similarity", state)
    
    async def _run_action_recommendation(self, state: MultiAgentState) -> MultiAgentState:
        """Run action recommendation agent"""
        return await self._execute_agent("recommendation", state)
    
    async def _run_data_storage(self, state: MultiAgentState) -> MultiAgentState:
        """Run data storage agent"""
        return await self._execute_agent("storage", state)
    
    async def _execute_agent(self, agent_name: str, state: MultiAgentState) -> MultiAgentState:
        """Execute a specific agent with error handling"""
        agent = self.agents[agent_name]
        start_time = datetime.now()
        
        try:
            # Log agent execution start
            self._log_execution(state, agent_name, "START", start_time)
            
            # Prepare agent input data
            agent_data = self._prepare_agent_data(agent_name, state)
            
            # Execute agent with timeout
            result = await agent.execute_with_timeout(
                agent_data, 
                settings.AGENT_TIMEOUT_SECONDS
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            if result["success"]:
                # Update state with agent results
                state = self._update_state_with_result(agent_name, state, result["result"])
                self._log_execution(state, agent_name, "SUCCESS", start_time, execution_time)
            else:
                # Handle agent failure
                error_msg = f"{agent_name} agent failed: {result.get('error', 'Unknown error')}"
                state["errors"].append(error_msg)
                self._log_execution(state, agent_name, "ERROR", start_time, execution_time, error_msg)
                logger.error(error_msg)
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = f"Exception in {agent_name} agent: {str(e)}"
            state["errors"].append(error_msg)
            self._log_execution(state, agent_name, "EXCEPTION", start_time, execution_time, error_msg)
            logger.error(error_msg)
        
        return state
    
    def _prepare_agent_data(self, agent_name: str, state: MultiAgentState) -> Dict[str, Any]:
        """Prepare input data for specific agent"""
        base_data = {"transaction": state["transaction"]}
        
        if agent_name == "duplicate":
            return base_data
        elif agent_name == "classification":
            return base_data
        elif agent_name == "similarity":
            return base_data
        elif agent_name == "recommendation":
            return {
                **base_data,
                "fraud_analysis": state.get("fraud_analysis", {}),
                "similar_cases": state.get("similarity_data", {}).get("similar_cases", [])
            }
        elif agent_name == "storage":
            return {
                **base_data,
                "fraud_analysis": state.get("fraud_analysis", {}),
                "similarity_data": state.get("similarity_data", {}),
                "recommendation": state.get("recommendation", {})
            }
        
        return base_data
    
    def _update_state_with_result(self, agent_name: str, state: MultiAgentState, 
                                 result: Dict[str, Any]) -> MultiAgentState:
        """Update state with agent execution results"""
        if agent_name == "duplicate":
            state["duplicate_check"] = result
        elif agent_name == "classification":
            state["fraud_analysis"] = result
        elif agent_name == "similarity":
            state["similarity_data"] = result
        elif agent_name == "recommendation":
            state["recommendation"] = result
        elif agent_name == "storage":
            state["storage_results"] = result
        
        return state
    
    def _should_continue_after_duplicate(self, state: MultiAgentState) -> str:
        """Determine if workflow should continue after duplicate check"""
        duplicate_check = state.get("duplicate_check", {})
        
        if duplicate_check.get("is_duplicate", False):
            state["workflow_status"] = "SKIPPED_DUPLICATE"
            return "skip"
        else:
            return "continue"
    
    def _log_execution(self, state: MultiAgentState, agent_name: str, status: str,
                      start_time: datetime, execution_time: float = None, error: str = None):
        """Log agent execution details"""
        log_entry = {
            "agent": agent_name,
            "status": status,
            "timestamp": start_time.isoformat(),
            "execution_time_seconds": execution_time
        }
        
        if error:
            log_entry["error"] = error
        
        state["execution_log"].append(log_entry)
    
    async def process_transaction(self, transaction: Transaction) -> MultiAgentState:
        """Process a transaction through the multi-agent workflow"""
        transaction_id = transaction.transaction_id
        
        # Initialize workflow state
        initial_state = MultiAgentState(
            transaction=transaction.model_dump(default=str),
            duplicate_check=None,
            fraud_analysis=None,
            similarity_data=None,
            recommendation=None,
            storage_results=None,
            workflow_status="PROCESSING",
            errors=[],
            execution_log=[]
        )
        
        # Track active execution
        self.active_executions[transaction_id] = {
            "start_time": datetime.now(),
            "status": "PROCESSING"
        }
        
        try:
            logger.info(f"Starting multi-agent workflow for transaction {transaction_id}")
            
            # Execute workflow
            final_state = await self.workflow.ainvoke(initial_state)
            
            # Update execution tracking
            execution_time = (datetime.now() - self.active_executions[transaction_id]["start_time"]).total_seconds()
            
            # Determine final status
            if final_state["errors"]:
                final_state["workflow_status"] = "COMPLETED_WITH_ERRORS"
            elif final_state.get("duplicate_check", {}).get("is_duplicate", False):
                final_state["workflow_status"] = "SKIPPED_DUPLICATE"
            else:
                final_state["workflow_status"] = "COMPLETED_SUCCESS"
            
            # Log completion
            logger.info(
                f"Multi-agent workflow completed for {transaction_id}: "
                f"status={final_state['workflow_status']}, "
                f"execution_time={execution_time:.2f}s, "
                f"errors={len(final_state['errors'])}"
            )
            
            return final_state
            
        except Exception as e:
            logger.error(f"Multi-agent workflow failed for {transaction_id}: {e}")
            initial_state["workflow_status"] = "FAILED"
            initial_state["errors"].append(f"Workflow exception: {str(e)}")
            return initial_state
            
        finally:
            # Clean up execution tracking
            if transaction_id in self.active_executions:
                del self.active_executions[transaction_id]
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get workflow execution statistics"""
        stats = {
            "active_executions": len(self.active_executions),
            "agent_statistics": {}
        }
        
        # Get statistics for each agent
        for name, agent in self.agents.items():
            agent_stats = agent.get_status()
            stats["agent_statistics"][name] = agent_stats
        
        return stats
    
    def shutdown_all_agents(self):
        """Shutdown all agents"""
        logger.info("Shutting down all agents in multi-agent workflow")
        for agent in self.agents.values():
            agent.shutdown()
