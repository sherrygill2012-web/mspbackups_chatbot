"""
Slack Bot Integration for MSP360 Backup Expert
Enables the chatbot to respond to queries in Slack channels and DMs
"""

import os
import re
import asyncio
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# Check if slack_bolt is available
try:
    from slack_bolt.async_app import AsyncApp
    from slack_bolt.adapter.socket_mode.async_handler import AsyncSocketModeHandler
    SLACK_AVAILABLE = True
except ImportError:
    SLACK_AVAILABLE = False
    print("Slack SDK not installed. Run: pip install slack-bolt")

from msp_expert import create_msp_expert


class MSP360SlackBot:
    """
    Slack bot for MSP360 Backup Expert.
    
    Responds to:
    - Direct messages
    - Mentions in channels
    - Specific keywords (optional)
    
    Required environment variables:
    - SLACK_BOT_TOKEN: Bot User OAuth Token (xoxb-...)
    - SLACK_APP_TOKEN: App-Level Token for Socket Mode (xapp-...)
    """
    
    def __init__(self):
        """Initialize the Slack bot"""
        if not SLACK_AVAILABLE:
            raise ImportError("slack-bolt package is required. Install with: pip install slack-bolt")
        
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        
        if not self.bot_token or not self.app_token:
            raise ValueError(
                "SLACK_BOT_TOKEN and SLACK_APP_TOKEN environment variables are required. "
                "See https://api.slack.com/start/building/bolt-python for setup instructions."
            )
        
        # Initialize Slack app
        self.app = AsyncApp(token=self.bot_token)
        
        # Initialize MSP360 agent
        self.agent = None
        self.deps = None
        
        # Register event handlers
        self._register_handlers()
    
    def _register_handlers(self):
        """Register Slack event handlers"""
        
        @self.app.event("app_mention")
        async def handle_mention(event, say, client):
            """Handle when the bot is mentioned in a channel"""
            await self._handle_message(event, say, client)
        
        @self.app.event("message")
        async def handle_dm(event, say, client):
            """Handle direct messages"""
            # Only respond to DMs (not channel messages without mention)
            if event.get("channel_type") == "im":
                await self._handle_message(event, say, client)
        
        @self.app.command("/msp360")
        async def handle_slash_command(ack, command, say):
            """Handle /msp360 slash command"""
            await ack()
            
            query = command.get("text", "").strip()
            if not query:
                await say(
                    "Please provide a question. Example: `/msp360 How to fix error code 1531?`"
                )
                return
            
            # Show typing indicator
            await say("🔍 Searching MSP360 documentation...")
            
            # Get response
            response = await self._get_agent_response(query)
            await say(response)
    
    async def _ensure_agent(self):
        """Ensure agent is initialized"""
        if self.agent is None:
            self.agent, self.deps = create_msp_expert()
    
    async def _get_agent_response(self, query: str) -> str:
        """
        Get response from the MSP360 agent.
        
        Args:
            query: User's question
        
        Returns:
            Formatted response for Slack
        """
        await self._ensure_agent()
        
        try:
            result = await self.agent.run(query, deps=self.deps)
            response = result.data
            
            # Convert markdown to Slack formatting
            response = self._convert_to_slack_format(response)
            
            return response
            
        except Exception as e:
            return f"❌ Error processing your question: {str(e)}"
    
    def _convert_to_slack_format(self, text: str) -> str:
        """
        Convert markdown to Slack mrkdwn format.
        
        Args:
            text: Markdown formatted text
        
        Returns:
            Slack mrkdwn formatted text
        """
        # Convert headers
        text = re.sub(r'^### (.+)$', r'*\1*', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', r'*\1*', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', r'*\1*', text, flags=re.MULTILINE)
        
        # Convert bold (already compatible)
        text = re.sub(r'\*\*(.+?)\*\*', r'*\1*', text)
        
        # Convert links [text](url) to <url|text>
        text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'<\2|\1>', text)
        
        # Convert inline code (already compatible with backticks)
        
        # Convert code blocks
        text = re.sub(r'```(\w+)?\n', r'```\n', text)
        
        return text
    
    async def _handle_message(self, event: dict, say, client):
        """
        Handle incoming message.
        
        Args:
            event: Slack event data
            say: Function to send response
            client: Slack client
        """
        # Get the message text
        text = event.get("text", "")
        
        # Remove bot mention if present
        text = re.sub(r'<@[A-Z0-9]+>', '', text).strip()
        
        if not text:
            await say(
                "Hi! I'm the MSP360 Backup Expert. Ask me anything about MSP360 Backup! "
                "For example: `How to fix error code 1531?`"
            )
            return
        
        # Check if it's a help request
        if text.lower() in ['help', 'hi', 'hello']:
            await say(self._get_help_message())
            return
        
        # Show typing indicator
        channel = event.get("channel")
        try:
            # Add reaction to show we're processing
            await client.reactions_add(
                channel=channel,
                name="eyes",
                timestamp=event.get("ts")
            )
        except:
            pass  # Ignore reaction errors
        
        # Get response from agent
        response = await self._get_agent_response(text)
        
        # Send response in thread if it's a channel message
        thread_ts = event.get("thread_ts") or event.get("ts")
        
        # Split long messages (Slack has 4000 char limit)
        if len(response) > 3900:
            chunks = self._split_message(response, 3900)
            for chunk in chunks:
                await say(text=chunk, thread_ts=thread_ts)
        else:
            await say(text=response, thread_ts=thread_ts)
        
        # Remove processing reaction
        try:
            await client.reactions_remove(
                channel=channel,
                name="eyes",
                timestamp=event.get("ts")
            )
            await client.reactions_add(
                channel=channel,
                name="white_check_mark",
                timestamp=event.get("ts")
            )
        except:
            pass
    
    def _split_message(self, text: str, max_length: int) -> list:
        """Split long message into chunks"""
        chunks = []
        current_chunk = ""
        
        for line in text.split('\n'):
            if len(current_chunk) + len(line) + 1 > max_length:
                chunks.append(current_chunk)
                current_chunk = line
            else:
                current_chunk += ('\n' if current_chunk else '') + line
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def _get_help_message(self) -> str:
        """Get help message"""
        return """
👋 *Hi! I'm the MSP360 Backup Expert Bot*

I can help you with:
• 🔧 Troubleshooting backup errors
• 🔢 Looking up error codes
• 📖 Explaining backup concepts
• ⚙️ Configuration guidance
• ✅ Best practices advice

*Example questions:*
• `How to fix error code 1531?`
• `What is synthetic full backup?`
• `How to configure Forever Forward Incremental?`
• `VSS error access denied`
• `What cloud storage supports synthetic backup?`

*Commands:*
• `/msp360 <question>` - Ask a question using slash command
• Mention me with `@MSP360 Expert` in any channel
• Send me a direct message

Let me know how I can help! 💾
"""
    
    async def start(self):
        """Start the Slack bot"""
        print("🚀 Starting MSP360 Backup Expert Slack Bot...")
        
        # Initialize agent
        await self._ensure_agent()
        print("✅ Agent initialized")
        
        # Start Socket Mode handler
        handler = AsyncSocketModeHandler(self.app, self.app_token)
        print("✅ Slack bot connected! Listening for messages...")
        await handler.start_async()


async def main():
    """Main entry point for the Slack bot"""
    bot = MSP360SlackBot()
    await bot.start()


if __name__ == "__main__":
    asyncio.run(main())

