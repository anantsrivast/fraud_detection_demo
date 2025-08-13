#!/usr/bin/env python3
"""
Main runner script for the fraud detection agents
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Fraud Detection Agent Runner')
    parser.add_argument('--agent', 
                       choices=['memory', 'memory-kafka', 'hybrid', 'hybrid-kafka'],
                       default='memory',
                       help='Which agent to run (default: memory)')
    parser.add_argument('--file', 
                       action='store_true',
                       help='Run file-based processing instead of Kafka')
    
    args = parser.parse_args()
    
    if args.agent == 'memory':
        if args.file:
            from agents.agent_with_mem import main as agent_main
        else:
            print("Memory agent requires --file flag for file-based processing")
            return
    elif args.agent == 'memory-kafka':
        from agents.agent_with_mem_kafka import main as agent_main
    elif args.agent == 'hybrid':
        from agents.hybrid_agent import main as agent_main
    elif args.agent == 'hybrid-kafka':
        from agents.hybrid_agent_kafka import main as agent_main
    else:
        print(f"Unknown agent: {args.agent}")
        return
    
    # Run the agent
    import asyncio
    asyncio.run(agent_main())

if __name__ == "__main__":
    main()
