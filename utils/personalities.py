import os

# Default personalities mapping
CHANNEL_PERSONALITIES = {
    int(os.getenv("SUPPORT_CHANNEL_ID", "0")): "Your name is Zade. You are a professional support agent. Be helpful, clear, and direct.",
    int(os.getenv("LOUNGE_CHANNEL_ID", "0")): "Your name is Zade. You are a witty and sarcastic community member. Keep things light but slightly edgy.",
}

DEFAULT_SYSTEM_PROMPT = "Your name is Zade. You are a helpful and chill community member. Be natural and conversational. Don't act like an AI or a bot unless asked."

def get_personality(bot, channel_id: int) -> str:
    """Retrieve personality from bot's persistent config or defaults."""
    # Convert channel_id to string for JSON compatibility if needed
    str_id = str(channel_id)
    override = bot.personality_overrides.get(str_id)
    if not override:
        # Check int for local defaults
        override = CHANNEL_PERSONALITIES.get(channel_id)
    
    return override or DEFAULT_SYSTEM_PROMPT

def set_personality_override(bot, channel_id: int, prompt: str):
    """Save personality override to bot's persistent config."""
    bot.personality_overrides[str(channel_id)] = prompt
    bot.save_persistent_config()
