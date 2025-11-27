#!/usr/bin/env python3
"""
CLI Interface for MSP360 Backup Expert Chatbot
Simple command-line interface for testing the agent
"""

import asyncio
import sys
from dotenv import load_dotenv

from msp_expert import create_msp_expert

load_dotenv()


async def main():
    """Main CLI loop"""
    print("=" * 80)
    print("💾 MSP360 Backup Expert Assistant - CLI")
    print("=" * 80)
    print("Ask questions about MSP360 Backup documentation.")
    print("Type 'exit' or 'quit' to end the session.")
    print("Type 'help' for usage tips.")
    print("=" * 80)
    print()
    
    # Initialize agent
    print("Initializing agent...")
    agent = None
    deps = None
    try:
        agent, deps = create_msp_expert()
        print("✓ Agent ready!")
        print()
    except Exception as e:
        print(f"❌ Error initializing agent: {e}")
        print("Make sure your .env file is configured with:")
        print("  - OPENAI_API_KEY (for LLM)")
        print("  - GEMINI_API_KEY (for embeddings)")
        print("  - QDRANT_URL")
        sys.exit(1)
    
    # Main loop
    try:
        await chat_loop(agent, deps)
    finally:
        # Cleanup: close Qdrant client if it exists
        if deps and hasattr(deps, 'qdrant_tools'):
            try:
                if hasattr(deps.qdrant_tools.client, 'close'):
                    deps.qdrant_tools.client.close()
            except:
                pass  # Ignore cleanup errors


async def chat_loop(agent, deps):
    """Chat loop separated for better cleanup handling"""
    while True:
        try:
            # Get user input
            query = input("\n💬 You: ").strip()
            
            if not query:
                continue
            
            # Check for exit commands
            if query.lower() in ['exit', 'quit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Check for help command
            if query.lower() == 'help':
                print_help()
                continue
            
            # Process query
            print("\n🤖 Assistant: ", end="", flush=True)
            
            try:
                result = await agent.run(query, deps=deps)
                print(result.data)
            except asyncio.CancelledError:
                print("\n❌ Request cancelled")
                continue
            except RuntimeError as e:
                if "Event loop is closed" in str(e):
                    print(f"\n❌ Error: {e}")
                    print("Event loop issue detected. Please restart the CLI.")
                    break
                else:
                    print(f"\n❌ Runtime Error: {e}")
                    print("Please try rephrasing your question.")
            except Exception as e:
                error_msg = str(e)
                print(f"\n❌ Error: {e}")
                if "event loop" not in error_msg.lower():
                    print("Please try rephrasing your question.")
        
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            continue


def print_help():
    """Print help information"""
    help_text = """
📚 MSP360 Backup Expert Assistant - Help

USAGE:
  Just type your question and press Enter!

EXAMPLE QUESTIONS:
  - How to fix error code 1531?
  - What is synthetic full backup?
  - Explain Forever Forward Incremental
  - How to troubleshoot VSS errors?
  - What storage providers support synthetic backup?
  - How to restore a backup plan?
  - Fix I/O error 1076
  - Configure backup retention policy

FEATURES:
  - Search MSP360 Backup documentation
  - Get solutions for specific error codes
  - Step-by-step troubleshooting guides
  - Configuration best practices
  - Cloud storage setup help
  - View source documentation URLs

TIPS:
  - Mention specific error codes for targeted results
  - Ask about backup concepts for detailed explanations
  - Request step-by-step guides for procedures
  - Follow-up questions are supported

COMMANDS:
  help  - Show this help message
  exit  - Exit the program (or Ctrl+C)
  quit  - Exit the program
"""
    print(help_text)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        sys.exit(0)

