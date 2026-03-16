import os
import json
import discord
from discord.ext import commands
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("DISCORD_TOKEN")
DEDICATED_CHANNELS = [int(id_.strip()) for id_ in os.getenv("DEDICATED_CHANNEL_IDS", "").split(",") if id_.strip()]
CONFIG_FILE = "config.json"

class AutonomousAgentBot(commands.Bot):
    def __init__(self):
        intents = discord.Intents.default()
        intents.message_content = True
        intents.members = True
        intents.voice_states = True
        intents.dm_messages = True
        super().__init__(command_prefix="!", intents=intents, help_command=None)
        self.channel_activity = {}  # Tracks timestamps for nudge system
        self.load_persistent_config()

    def load_persistent_config(self):
        """Load toggles and channels from JSON with hardcoded defaults."""
        if os.path.exists(CONFIG_FILE):
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                self.auto_msg_enabled = data.get("auto_msg_enabled", False)
                self.auto_reply_enabled = data.get("auto_reply_enabled", True)
                self.auto_reply_channels = set(int(id_) for id_ in data.get("auto_reply_channels", []))
                self.respected_ids = set(int(id_) for id_ in data.get("respected_ids", []))
                self.personality_overrides = data.get("personality_overrides", {})
                self.tasks = data.get("tasks", [])
                self.user_relationships = data.get("user_relationships", {})
                self.server_culture = data.get("server_culture", {"slang": [], "trending_topics": []})
                self.game_states = data.get("game_states", {})
                self.user_age_verified = data.get("user_age_verified", {})
        else:
            # First-run defaults
            self.auto_msg_enabled = False
            self.auto_reply_enabled = True
            self.auto_reply_channels = set()
            self.respected_ids = set()
            self.personality_overrides = {}
            self.tasks = []
            self.user_relationships = {}
            self.server_culture = {"slang": [], "trending_topics": []}
            self.game_states = {}
            self.user_age_verified = {}
            self.save_persistent_config()

    def save_persistent_config(self):
        """Save current state to JSON."""
        with open(CONFIG_FILE, "w") as f:
            json.dump({
                "auto_msg_enabled": self.auto_msg_enabled,
                "auto_reply_enabled": self.auto_reply_enabled,
                "auto_reply_channels": list(self.auto_reply_channels),
                "respected_ids": list(self.respected_ids),
                "personality_overrides": self.personality_overrides,
                "tasks": self.tasks,
                "user_relationships": self.user_relationships,
                "server_culture": self.server_culture,
                "game_states": self.game_states,
                "user_age_verified": self.user_age_verified
            }, f, indent=4)


    async def setup_hook(self):
        # ... (rest of setup_hook)
        await self.load_extension("cogs.ai_agent")
        print("Cogs loaded.")
        await self.tree.sync()
        print("Slash commands synced.")

    async def on_ready(self):
        # ... (on_ready logic)
        print(f"Logged in as {self.user} (ID: {self.user.id})")
        print("------")
        
        # Send auto-greeting if enabled
        if self.auto_msg_enabled:
            content = os.getenv("AUTO_MESSAGE_CONTENT", "Hello!")
            for channel_id in self.auto_reply_channels:
                channel = self.get_channel(channel_id)
                if channel:
                    try:
                        await channel.send(content)
                        print(f"Auto-greeting sent to #{channel.name}")
                    except Exception as e:
                        print(f"Failed to send to {channel_id}: {e}")

    async def on_message(self, message: discord.Message):
        # Ignore logic: strictly ignore all other bots and its own messages
        if message.author.bot:
            return

        ai_cog = self.get_cog("AIAgent")
        if ai_cog:
            from cogs.ai_agent import IGNORED_USERS
            if message.author.id in IGNORED_USERS:
                return

        # --- AI-Moderation Nudge System ---
        # Track activity to detect heated moments (e.g., 5 messages in 10 seconds)
        import time
        now = time.time()
        chan_id = message.channel.id
        
        if chan_id not in self.channel_activity:
            self.channel_activity[chan_id] = []
        self.channel_activity[chan_id].append(now)
        
        # Keep only the last 5 messages' timestamps
        if len(self.channel_activity[chan_id]) > 5:
            self.channel_activity[chan_id].pop(0)

        # If 5 messages were sent in less than 10 seconds, trigger a nudge check
        if len(self.channel_activity[chan_id]) == 5 and (now - self.channel_activity[chan_id][0]) < 10:
            self.channel_activity[chan_id].clear() # Reset to avoid spam
            if ai_cog:
                self.loop.create_task(ai_cog.check_and_nudge(message.channel))

        # --- Dual-Trigger System ---
        is_dm = isinstance(message.channel, discord.DMChannel)
        is_mentioned = self.user.mentioned_in(message)
        is_auto_reply_channel = message.channel.id in self.auto_reply_channels
        
        # New: Check if the message is a reply to the bot
        is_reply_to_me = False
        if message.reference and message.reference.message_id:
            if message.reference.resolved and isinstance(message.reference.resolved, discord.Message):
                if message.reference.resolved.author.id == self.user.id:
                    is_reply_to_me = True

        should_trigger = is_dm or is_mentioned or is_reply_to_me or (self.auto_reply_enabled and is_auto_reply_channel)
        
        # Don't trigger AI if it's a bot command
        if message.content.startswith(self.command_prefix):
            should_trigger = False

        if should_trigger:
            content = message.content
            if is_mentioned:
                content = content.replace(f"<@{self.user.id}>", "").replace(f"<@!{self.user.id}>", "").strip()
            
            if ai_cog:
                await ai_cog.process_ai_request(message, content)

        # Allow commands to still work
        await self.process_commands(message)

if __name__ == "__main__":
    bot = AutonomousAgentBot()
    bot.run(TOKEN)
