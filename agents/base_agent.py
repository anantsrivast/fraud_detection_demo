from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import asyncio
import logging
from datetime import datetime, timedelta
import uuid

logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    def __init__(self, agent_id: Optional[str] = None):
        self.agent_id = agent_id or f"{self.__class__.__name__}_{uuid.uuid4().hex[:8]}"
        self.created_at = datetime.now()
        self.last_activity = datetime.now()
        self.processed_count = 0
        self.error_count = 0
        self.is_active = True
    
    @abstractmethod
    async def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data and return results"""
        pass
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status information"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.__class__.__name__,
            "is_active": self.is_active,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "processed_count": self.processed_count,
            "error_count": self.error_count,
            "uptime_seconds": (datetime.now() - self.created_at).total_seconds()
        }
    
    async def execute_with_timeout(self, data: Dict[str, Any], timeout_seconds: int = 30) -> Dict[str, Any]:
        """Execute agent processing with timeout"""
        try:
            self.last_activity = datetime.now()
            
            # Execute with timeout
            result = await asyncio.wait_for(
                self.process(data), 
                timeout=timeout_seconds
            )
            
            self.processed_count += 1
            self.last_activity = datetime.now()
            
            return {
                "success": True,
                "agent_id": self.agent_id,
                "result": result,
                "execution_time": (datetime.now() - self.last_activity).total_seconds()
            }
            
        except asyncio.TimeoutError:
            self.error_count += 1
            logger.error(f"Agent {self.agent_id} timed out after {timeout_seconds} seconds")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": "Timeout",
                "error_type": "TimeoutError"
            }
        except Exception as e:
            self.error_count += 1
            logger.error(f"Agent {self.agent_id} error: {e}")
            return {
                "success": False,
                "agent_id": self.agent_id,
                "error": str(e),
                "error_type": type(e).__name__
            }
    
    def shutdown(self):
        """Shutdown the agent"""
        self.is_active = False
        logger.info(f"Agent {self.agent_id} shutdown")
