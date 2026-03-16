# 🤖 Autonomous Discord AI Agent

A state-of-the-art Discord bot utilizing **LangGraph** for advanced agentic reasoning, **OpenRouter** for LLM access, and **ChromaDB** for persistent long-term memory.

## 🚀 Advanced Features

- **🧠 Agentic Reasoning (LangGraph)**: Unlike basic bots, this agent uses a state-graph to decide when to use tools, when to summarize conversation history, and how to maintain context across long sessions.
- **⚡ Dual-Trigger System**:
  - **Slash Commands**: Modern, discoverable UI commands.
  - **Prefix Commands**: Traditional `!` commands for power users.
  - **Conversational Triggers**: Automatically replies to mentions, replies to its own messages, and DMs.
- **💾 Evolving Memory**:
  - **Vector Store**: Remembers your preferences and past conversations using ChromaDB.
  - **Vibe System**: Tracks a "Relationship Score" with every user that influences the bot's tone and helpfulness.
- **🎵 Multimedia & Vision**:
  - **Vision Support**: Can "see" and describe images you upload.
  - **Music System**: Full voice-channel integration with YouTube/SoundCloud support.
  - **Image Generation**: Built-in DALL-E/Flux support with automatic NSFW filtering based on channel settings.
- **🛡️ Proactive Safety**:
  - **Nudge System**: Detects heated arguments (5 messages in 10s) and attempts to de-escalate.
  - **Age Verification**: One-time mandatory verification for DM users to control NSFW content access.

## 🛠️ Command Reference

### User Commands
| Slash | Prefix | Description |
|-------|--------|-------------|
| `/help` | `!help` | Interactive help menu |
| `/recap` | `!recap` | AI summary of the last 100 messages |
| `/ping` | `!ping` | Check bot latency |
| `/status` | `!status` | Check system health |
| `/ask_memory` | - | Query what the bot remembers about you |
| `/vibe_with_me` | - | Check your relationship score |

### Admin & Staff
- `/auto_reply <add/remove/list>`: Manage channels where the bot replies automatically.
- `/personality <persona>`: Override the bot's character for the current channel.
- `/vibe_check`: Analyze the overall sentiment of recent server messages.
- `!sync`: Force-synchronize all slash commands with Discord.
- `/set_model`: Switch AI providers or models (e.g., GPT-4o to Gemini) at runtime.

## 🏗️ Architecture

The bot's "brain" is a **LangGraph State Machine**:
1. **Input**: Receives message + Vision data + User Relationship score.
2. **Node: Agent**: Decides if a tool (Music, Image Gen, Memory) is needed.
3. **Node: Tools**: Executes requested actions and returns results to the agent.
4. **Node: Summarize**: Every 15 messages, the agent automatically "compresses" its memory to stay fast and token-efficient.
5. **Output**: Sends a natural, contextual response.

## ⚙️ Configuration

The bot uses a hybrid configuration system:
- **`.env`**: Stores sensitive API keys and core model selections.
- **`config.json`**: Stores all runtime settings (auto-reply channels, relationship scores, culture trends). **This file is updated automatically by the bot.**

### Initial Setup
1. `pip install -r requirements.txt`
2. Configure `.env` with your `DISCORD_TOKEN` and `OPENROUTER_API_KEY`.
3. Start the bot: `python main.py`.
4. Use `/auto_reply add` in a channel to enable automatic chat.

## 🔐 Privacy
- Conversations are stored locally in `chroma_db/` for memory features.
- User data is never sent to third parties except for the AI processing (OpenRouter/OpenAI).

## 📜 Credits
- AI was somewhat used to assist in the creation and optimization of several features within this project.
