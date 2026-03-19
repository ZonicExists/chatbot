import os
import asyncio
import base64
import aiohttp
import discord
import json
import logging
import datetime
import io
import re
from typing import TypedDict, Annotated, Sequence, List, Optional
from discord.ext import commands, tasks
from discord import app_commands, File

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from utils.personalities import get_personality, set_personality_override
from utils.tools import get_tools
from utils.vector_store import vector_store

# Set to store ignored user IDs
IGNORED_USERS = set()

# Admin IDs from .env
ADMIN_IDS = [int(id_.strip()) for id_ in os.getenv("ADMIN_USER_IDS", "").split(",") if id_.strip()]

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AIAgent")

# Suppress noisy third-party logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("google_genai").setLevel(logging.WARNING)
logging.getLogger("langchain_google_genai").setLevel(logging.WARNING)

def is_admin():
    async def predicate(interaction: discord.Interaction):
        if interaction.user.id in ADMIN_IDS:
            return True
        await interaction.response.send_message("❌ No auth.", ephemeral=True)
        return False
    return app_commands.check(predicate)

# Define Agent State
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], lambda x, y: x + y]
    base_prompt: str
    context_id: str
    user_id: str
    guild_id: str
    user_name: str
    summary: str 

class AIAgent(commands.Cog):
    def __init__(self, bot):
        self.bot = bot
        self.provider = os.getenv("AI_PROVIDER", "openrouter").lower()
        self.model_name = os.getenv("AI_MODEL_NAME", "openrouter/healer-alpha")
        self.llm = self._init_llm()
        self.tools = get_tools(bot)  
        self.cooldowns = commands.CooldownMapping.from_cooldown(1, 5.0, commands.BucketType.user)
        self.graph = self._build_graph()

    def _init_llm(self):
        try:
            if self.provider == "openai":
                return ChatOpenAI(model=self.model_name, api_key=os.getenv("OPENAI_API_KEY"), temperature=0.7)
            elif self.provider == "google" or self.provider == "local":
                return ChatGoogleGenerativeAI(model=self.model_name, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0.7)
            elif self.provider == "openrouter":
                return ChatOpenAI(
                    model=self.model_name, 
                    api_key=os.getenv("OPENROUTER_API_KEY"), 
                    base_url="https://openrouter.ai/api/v1",
                    default_headers={
                        "HTTP-Referer": "https://github.com/google/gemini-cli", # Required for OpenRouter
                        "X-Title": "Autonomous Discord AI Agent"
                    },
                    temperature=0.7
                )
            elif self.provider == "pollinations":
                return ChatOpenAI(
                    model=self.model_name,
                    api_key=os.getenv("POLLINATIONS_API_KEY"),
                    base_url="https://gen.pollinations.ai/v1",
                    temperature=0.7
                )
            else:
                return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))
        except Exception as e:
            logger.error(f"LLM Error: {e}")
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=os.getenv("GOOGLE_API_KEY"))

    def _build_graph(self):
        # OpenRouter models often need bind_tools to work with LangGraph
        try:
            model_with_tools = self.llm.bind_tools(self.tools)
            logger.info(f"Successfully bound {len(self.tools)} tools to the model.")
        except Exception as e:
            logger.error(f"CRITICAL: Error binding tools: {e}. Falling back to toolless model.")
            # Fallback for models that don't support tool calling directly
            model_with_tools = self.llm

        checkpointer = MemorySaver()

        async def summarize_context(state: AgentState):
            messages = state['messages']
            # We summarize periodically to save tokens and maintain speed
            logger.info("Summarizing context internally...")
            
            # Use only the last 20 messages for summary to keep it relevant and small
            # We also sanitize and filter to ensure a valid role sequence for the provider
            llm_messages = []
            last_role = None
            
            for msg in list(messages[-20:]):
                # Strictly convert content to plain text for summarization
                # This ensures NO image data/blocks are sent to the LLM
                content = msg.content
                text_content = ""

                if isinstance(content, str):
                    # Strip base64 if it somehow leaked into a string
                    text_content = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+', '[Image]', content)
                elif isinstance(content, list):
                    parts = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            parts.append(item.get("text", ""))
                        elif isinstance(item, dict) and item.get("type") == "image_url":
                            parts.append("[Image]")
                        else:
                            parts.append(str(item))
                    text_content = " ".join(parts)
                else:
                    text_content = str(content)

                if isinstance(msg, HumanMessage): 
                    role = "user"
                    new_msg = HumanMessage(content=text_content)
                elif isinstance(msg, AIMessage): 
                    role = "assistant"
                    # Use text_content if available, otherwise use a placeholder for tool calls
                    display_content = text_content
                    if not display_content and hasattr(msg, "tool_calls") and msg.tool_calls:
                        tool_names = [tc.get("name", "unknown") for tc in msg.tool_calls]
                        display_content = f"[Called tools: {', '.join(tool_names)}]"
                    
                    if not display_content:
                        continue
                    new_msg = AIMessage(content=display_content)
                else: 
                    # Skip SystemMessage and ToolMessage for summary
                    continue

                if role == "user" and last_role == "user":
                    if llm_messages:
                        llm_messages[-1].content += f"\n{text_content}"
                    continue
                
                llm_messages.append(new_msg)
                last_role = role

            # Ensure the sequence starts with a User message (required by most providers)
            while llm_messages and not isinstance(llm_messages[0], HumanMessage):
                llm_messages.pop(0)

            if not llm_messages:
                return {"summary": state.get("summary", "")}

            summary_prompt = (
                "Summarize the conversation so far into 2 sentences max. "
                f"Previous summary: {state.get('summary', 'None')}"
            )
            try:
                response = await self.llm.ainvoke([SystemMessage(content=summary_prompt)] + llm_messages)
                return {"summary": response.content} 
            except Exception as e:
                logger.error(f"Summarization Error: {e}")
                return {"summary": state.get('summary', 'Error summarizing context.')}

        async def call_model(state: AgentState):
            messages = list(state['messages'])
            
            # --- OPENROUTER / LLAMA / GEMINI FIX: Role sequence handling & Content Sanitization ---
            llm_messages = []
            last_role = None
            for msg in messages:
                # Sanitize content for LLM call WITHOUT modifying the original message object in state
                content = msg.content
                sanitized_content = content
                
                if isinstance(content, str) and len(content) > 10000:
                    if "data:image" in content and "base64" in content:
                        sanitized_content = content[:200] + "... [TRUNCATED BASE64 IMAGE DATA FOR CONTEXT] ..."
                elif isinstance(content, list):
                    sanitized_content = []
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            text = item.get("text", "")
                            if len(text) > 10000 and "data:image" in text:
                                sanitized_content.append({"type": "text", "text": text[:200] + "... [TRUNCATED BASE64 IMAGE DATA FOR CONTEXT] ..."})
                            else:
                                sanitized_content.append(item)
                        else:
                            sanitized_content.append(item)

                if isinstance(msg, HumanMessage): 
                    role = "user"
                    new_msg = HumanMessage(content=sanitized_content, name=getattr(msg, "name", None))
                elif isinstance(msg, AIMessage): 
                    role = "assistant"
                    new_msg = AIMessage(content=sanitized_content, tool_calls=getattr(msg, "tool_calls", []))
                elif isinstance(msg, SystemMessage): continue
                elif isinstance(msg, ToolMessage): 
                    role = "tool"
                    new_msg = ToolMessage(content=sanitized_content, tool_call_id=msg.tool_call_id)
                else: 
                    role = "user"
                    new_msg = HumanMessage(content=sanitized_content)

                if role == "user" and last_role == "user":
                    if llm_messages:
                        prev_content = llm_messages[-1].content
                        curr_content = sanitized_content
                        
                        # Handle merging
                        if isinstance(prev_content, str) and isinstance(curr_content, str):
                            llm_messages[-1].content += f"\n{curr_content}"
                        else:
                            if isinstance(prev_content, str): prev_content = [{"type": "text", "text": prev_content}]
                            if isinstance(curr_content, str): curr_content = [{"type": "text", "text": curr_content}]
                            llm_messages[-1].content = prev_content + curr_content
                    continue
                
                llm_messages.append(new_msg)
                last_role = role

            # HARD TRIM for LLM context: Keep only last 40 messages for longer memory
            if len(llm_messages) > 40:
                llm_messages = llm_messages[-40:]
            
            # Ensure it starts with a user message (required by many providers)
            while llm_messages and not isinstance(llm_messages[0], HumanMessage):
                llm_messages.pop(0)
            
            context_id = state['context_id']
            user_id = state['user_id']
            guild_id = state.get('guild_id', '0')
            
            rel = self.bot.user_relationships.get(user_id, {"score": 0})
            internal_context = f"\n[INTERNAL DATA]: You are speaking to {state['user_name']} (Relationship Score: {rel['score']}) in the server '{self.bot.get_guild(int(guild_id)).name if self.bot.get_guild(int(guild_id)) else 'Unknown Server'}'. "
            internal_context += "You MAY mention the user's name, server name, and channel name naturally. However, NEVER mention the Relationship Score or any raw Discord IDs."
            culture_context = f"\n[COMMUNITY CULTURE]: Trending topics: {', '.join(self.bot.server_culture['trending_topics'][:3])}."
            culture_context += " DO NOT mention these trending topics or slang unless specifically asked about them or if they are directly relevant to the user's query."
            
            tool_instructions = f"\n[TOOL USAGE]: When a user asks for server stats, play music, autoplay recommendations, image generation, or check memories, you MUST call the corresponding tool immediately. "
            tool_instructions += f"Use user_id='{user_id}' and guild_id='{guild_id}' for these tools. Do NOT ask for IDs. "
            tool_instructions += "CRITICAL: If you call `generate_image`, the image is handled automatically by the system. NEVER include raw 'data:image' strings, base64 data, or the 'Generated Image: ...' result in your final response. Just confirm the image was generated naturally."
            
            game_context = ""
            if context_id in self.bot.game_states:
                gs = self.bot.game_states[context_id]
                game_context = f"\n[GAME SESSION]: {gs['type']}. Current Data: {json.dumps(gs['data'])}. Rules: Update state after moves, reset on win."

            full_system_prompt = "Your name is Zade. " + state['base_prompt'] + internal_context + culture_context + tool_instructions + game_context
            if state.get('summary'):
                full_system_prompt += f"\n[PREVIOUS CONVERSATION SUMMARY]: {state['summary']}"

            # Always ensure SystemMessage is at the top
            final_to_send = [SystemMessage(content=full_system_prompt)] + llm_messages
            
            logger.info(f"--- Thinking (Msgs: {len(final_to_send)}) ---")
            
            # Use ainvoke for speed
            response = await model_with_tools.ainvoke(final_to_send)
            return {"messages": [response]}

        workflow = StateGraph(AgentState)
        workflow.add_node("agent", call_model)
        workflow.add_node("tools", ToolNode(self.tools))
        workflow.add_node("summarize", summarize_context)
        workflow.add_edge(START, "agent")

        def should_continue(state: AgentState):
            messages = state['messages']
            last_message = messages[-1]
            if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
                return "tools"
            # Summarize only every 10 messages to reduce latency from extra LLM calls
            if len(messages) > 15 and len(messages) % 10 == 0:
                return "summarize"
            return END

        workflow.add_conditional_edges("agent", should_continue)
        workflow.add_edge("tools", "agent")
        workflow.add_edge("summarize", END)
        return workflow.compile(checkpointer=checkpointer)

    async def _handle_vision(self, message: discord.Message) -> List[dict]:
        content = []
        if not message.attachments: return content
        for attachment in message.attachments:
            if any(attachment.filename.lower().endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
                try:
                    async with aiohttp.ClientSession() as session:
                        async with session.get(attachment.url) as resp:
                            if resp.status == 200:
                                image_data = await resp.read()
                                base64_image = base64.b64encode(image_data).decode("utf-8")
                                content.append({
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                                })
                except: pass
        return content

    def _get_clean_text(self, content) -> str:
        if isinstance(content, str): return content
        if isinstance(content, list):
            return "".join([p.get("text", "") if isinstance(p, dict) else str(p) for p in content])
        return str(content)

    async def _send_split_message(self, target, content: str, prefix: str = ""):
        if not content: return
        # Clean any accidental base64 leaks from the LLM
        content = re.sub(r'data:image/[^;]+;base64,[a-zA-Z0-9+/=]+', '[Image Data]', content)
        
        full_text = prefix + content
        if len(full_text) <= 2000:
            if isinstance(target, discord.Interaction): await target.followup.send(full_text)
            else: await target.reply(full_text)
        else:
            for i in range(0, len(full_text), 1900):
                chunk = full_text[i:i+1900]
                if isinstance(target, discord.Interaction): await target.followup.send(chunk)
                else: await target.channel.send(chunk)

    # --- Age Verification View for DMs ---
    class AgeVerificationView(discord.ui.View):
        def __init__(self, cog, user_id):
            super().__init__(timeout=None) # Permanent view until clicked
            self.cog = cog
            self.user_id = user_id

        @discord.ui.button(label="I am 18+ (Enable NSFW)", style=discord.ButtonStyle.success)
        async def verified_18(self, interaction: discord.Interaction, button: discord.ui.Button):
            if str(interaction.user.id) != self.user_id: return
            self.cog.bot.user_age_verified[self.user_id] = True
            self.cog.bot.save_persistent_config()
            await interaction.response.send_message("✅ Age verified. NSFW content is now **ENABLED** in DMs. You cannot change this setting later.", ephemeral=True)
            self.stop()

        @discord.ui.button(label="I am under 18 (SFW Only)", style=discord.ButtonStyle.danger)
        async def verified_sfw(self, interaction: discord.Interaction, button: discord.ui.Button):
            if str(interaction.user.id) != self.user_id: return
            self.cog.bot.user_age_verified[self.user_id] = False
            self.cog.bot.save_persistent_config()
            await interaction.response.send_message("🛡️ Age verified. NSFW content is now **RESTRICTED** in DMs. You cannot change this setting later.", ephemeral=True)
            self.stop()

    async def process_ai_request(self, message: discord.Message, text_content: str):
        bucket = self.cooldowns.get_bucket(message)
        if bucket.update_rate_limit(): return await message.channel.send("⚠️ Slow down!", delete_after=3)

        text_content = " ".join(text_content.split())[:1000]
        user_id_str = str(message.author.id)
        # DM Support: Use 'dm' as guild_id and simplify context_id
        is_dm = message.guild is None
        guild_id_str = str(message.guild.id) if not is_dm else "dm"
        chan_id_str = str(message.channel.id)
        context_id = f"{chan_id_str}-{user_id_str}" if not is_dm else f"dm-{user_id_str}"

        # --- ONE-TIME AGE VERIFICATION FOR DMs ---
        if is_dm and user_id_str not in self.bot.user_age_verified:
            embed = discord.Embed(
                title="🔞 Age Verification Required",
                description=(
                    "Before we continue in DMs, please verify your age status.\n\n"
                    "**Why?** This determines whether I can generate NSFW content or images for you here.\n"
                    "- Selecting **18+** enables unrestricted content.\n"
                    "- Selecting **Under 18** restricts all adult content.\n\n"
                    "⚠️ **Note:** This is a permanent setting for this account and cannot be changed later."
                ),
                color=discord.Color.gold()
            )
            view = self.AgeVerificationView(self, user_id_str)
            return await message.reply(embed=embed, view=view)

        if user_id_str not in self.bot.user_relationships:
            self.bot.user_relationships[user_id_str] = {"score": 0}
        self.bot.user_relationships[user_id_str]["score"] += 1
        self.bot.save_persistent_config()

        base_personality = get_personality(self.bot, message.channel.id)
        
        # --- Restore Respected User Logic ---
        is_lounge = message.channel.id == int(os.getenv("LOUNGE_CHANNEL_ID", 0))
        if is_lounge:
            if message.author.id in self.bot.respected_ids:
                # Override the witty persona with a polite one to ensure it's effective
                base_personality = "Your name is Zade. You are a helpful and polite community member. You are speaking to a respected authority figure, so be courteous and professional. Avoid sarcasm."
                logger.info(f"Respected user {message.author.display_name} ({message.author.id}) recognized in Lounge.")
            else:
                base_personality += "\n[NORMAL MODE: Use witty/sarcastic lounge persona.]"

        vision_content = await self._handle_vision(message)

        # Check if channel is NSFW (DMs use the saved verification status)
        if is_dm:
            is_nsfw_channel = self.bot.user_age_verified.get(user_id_str, False)
        else:
            is_nsfw_channel = getattr(message.channel, "nsfw", False)
            
        nsfw_context = "\n[CHANNEL SETTINGS]: This channel is NSFW-enabled. You may generate NSFW images if the user requests them by setting is_nsfw=True in the generate_image tool." if is_nsfw_channel else "\n[CHANNEL SETTINGS]: This channel is SFW. You MUST NOT generate NSFW content. Always keep is_nsfw=False in the generate_image tool."

        # If it's a multimodal message with no text, provide a placeholder for the context
        display_text = text_content
        if not display_text:
            if vision_content: display_text = "[Image Attachment]"
            else: display_text = "[Empty Message]"

        human_content = [{"type": "text", "text": f"[{message.author.display_name}]: {display_text}"}] + vision_content

        # Run vector store addition in the background to avoid blocking the LLM request
        asyncio.create_task(vector_store.add_messages([{
            "id": str(message.id),
            "content": display_text,
            "author": str(message.author.display_name),
            "author_id": user_id_str,
            "timestamp": message.created_at.isoformat()
        }]))

        inputs = {
            "messages": [HumanMessage(content=human_content)],
            "base_prompt": base_personality + f"\nContext: {context_id}." + nsfw_context,
            "context_id": context_id,
            "user_id": user_id_str,
            "guild_id": str(message.guild.id) if message.guild else "0",
            "user_name": message.author.display_name
        }

        async with message.channel.typing():
            try:
                config = {"configurable": {"thread_id": context_id}}
                final_state = await self.graph.ainvoke(inputs, config=config)
                
                # --- FIX: Initialize variables before searching the state ---
                final_response = ""
                image_file = None
                
                # Only scan messages from the CURRENT turn
                for msg in reversed(list(final_state['messages'])):
                    if isinstance(msg, HumanMessage):
                        break # Stop at the user's message; don't process old turns
                    
                    if isinstance(msg, AIMessage) and msg.content and not final_response:
                        final_response = self._get_clean_text(msg.content)
                    
                    if isinstance(msg, ToolMessage) and "Generated Image: data:image" in msg.content and not image_file:
                        try:
                            # Extract base64
                            b64_match = re.search(r'data:image/([^;]+);base64,([a-zA-Z0-9+/]+={0,2})', msg.content)
                            if b64_match:
                                ext = b64_match.group(1)
                                b64_data = b64_match.group(2).strip()
                                missing_padding = len(b64_data) % 4
                                if missing_padding:
                                    b64_data += '=' * (4 - missing_padding)
                                image_bytes = base64.b64decode(b64_data)
                                image_file = File(io.BytesIO(image_bytes), filename=f"generated_image.{ext}")
                                
                                # Memory optimization: Replace the massive base64 in state
                                msg.content = f"Generated Image: [File sent as {ext}. Base64 removed.]"
                        except Exception as e:
                            logger.error(f"Error decoding image in chat: {e}")
                
                if final_response:
                    if image_file:
                        # Combine text and image into a single embed
                        embed = discord.Embed(description=final_response, color=discord.Color.random())
                        embed.set_image(url=f"attachment://{image_file.filename}")
                        await message.reply(embed=embed, file=image_file)
                    else:
                        await self._send_split_message(message, final_response)
                elif image_file:
                    await message.reply(file=image_file)
                else:
                    logger.warning("No AIMessage found in final state.")
            except Exception as e:
                logger.error(f"Thinking Error: {e}")
                await message.reply("❌ Error thinking.")

    # --- Commands ---
    @app_commands.command(name="start_game", description="Start game.")
    async def start_game(self, interaction: discord.Interaction, game_type: str):
        context_id = f"{interaction.channel.id}-{interaction.user.id}"
        self.bot.game_states[context_id] = {
            "active": True, "type": game_type, 
            "data": {"board": [[" "," "," "],[" "," "," "],[" "," "," "]], "players": [interaction.user.display_name]}
        }
        self.bot.save_persistent_config()
        await interaction.response.send_message(f"🎮 Started! Talk to play.")

    @app_commands.command(name="vibe_with_me", description="Check friendship.")
    async def vibe_with_me(self, interaction: discord.Interaction):
        score = self.bot.user_relationships.get(str(interaction.user.id), {"score": 0})["score"]
        await interaction.response.send_message(f"💖 Score: {score}")

    @app_commands.command(name="personality", description="Set personality.")
    @is_admin()
    async def personality(self, interaction: discord.Interaction, persona: str):
        set_personality_override(self.bot, interaction.channel.id, persona)
        await interaction.response.send_message(f"🎭 Updated.")

    @app_commands.command(name="recap", description="100-msg summary.")
    async def recap(self, interaction: discord.Interaction):
        await interaction.response.defer()
        history = []
        async for msg in interaction.channel.history(limit=100):
            if not msg.author.bot: history.append(f"{msg.author.name}: {msg.content}")
        if not history: return await interaction.followup.send("No data.")
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Your name is Zade. Summarize the following conversation in 3 concise sentences."), HumanMessage(content="\n".join(history[:40]))])
            await self._send_split_message(interaction, self._get_clean_text(response.content), prefix="### 📋 Recap\n")
        except: await interaction.followup.send("Failed.")

    @app_commands.command(name="ask_memory", description="Query memory.")
    async def ask_memory(self, interaction: discord.Interaction, query: str):
        await interaction.response.defer()
        context = vector_store.query(query, n_results=2, filter_dict={"author_id": str(interaction.user.id)})
        if not context: return await interaction.followup.send("❌ No memories.")
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Your name is Zade. Use the following context from your memory to answer the user's query naturally."), HumanMessage(content=f"Query: {query}\n\nContext: {context}")])
            await self._send_split_message(interaction, self._get_clean_text(response.content), prefix=f"### 🧠 Memory\n")
        except: await interaction.followup.send("Failed.")

    @app_commands.command(name="status", description="Bot status.")
    async def status(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"🤖 Bot ok. Latency: {round(self.bot.latency * 1000)}ms")

    @app_commands.command(name="set_model", description="Switch AI.")
    @is_admin()
    async def set_model(self, interaction: discord.Interaction, provider: str, model_name: str):
        self.provider, self.model_name = provider.lower(), model_name
        self.llm = self._init_llm()
        self.tools = get_tools(self.bot)
        self.graph = self._build_graph()
        await interaction.response.send_message(f"✅ Updated to {provider}.")

    @app_commands.command(name="server_report", description="Server report.")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def server_report(self, interaction: discord.Interaction):
        if not interaction.guild: return await interaction.response.send_message("❌ Server only.", ephemeral=True)
        await interaction.response.defer()
        prompt = f"Summarize report. Stats: {interaction.guild.name}, {interaction.guild.member_count} members."
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Your name is Zade and you are a data analyst. Provide a professional summary of the server report."), HumanMessage(content=prompt)])
            await self._send_split_message(interaction, self._get_clean_text(response.content), prefix="### 📊 Report\n")
        except: await interaction.followup.send("Failed.")

    @app_commands.command(name="ping", description="Pong!")
    async def ping(self, interaction: discord.Interaction):
        await interaction.response.send_message(f"🏓 {round(self.bot.latency * 1000)}ms")

    # --- Music Commands ---
    def _create_bot_embed(self, title: str, description: str, color: discord.Color = discord.Color.blue(), footer: str = "🎵 Music System"):
        embed = discord.Embed(title=title, description=description, color=color, timestamp=datetime.datetime.now(datetime.timezone.utc))
        embed.set_footer(text=footer)
        return embed

    # --- Music View ---
    class MusicControlView(discord.ui.View):
        def __init__(self, cog, user_id):
            super().__init__(timeout=120)
            self.cog = cog
            self.user_id = user_id

        async def _run_tool(self, interaction, tool_name, **kwargs):
            if str(interaction.user.id) != self.user_id:
                return await interaction.response.send_message("❌ Not your session.", ephemeral=True)
            
            tool = next((t for t in self.cog.tools if t.name == tool_name), None)
            if not tool: return await interaction.response.send_message("❌ Tool not found.", ephemeral=True)
            
            await interaction.response.defer()
            kwargs['user_id'] = self.user_id
            
            try:
                # Use ainvoke for all tools as it handles both sync and async internally in LangChain 0.2+
                res = await tool.ainvoke(kwargs)
            except Exception as e:
                res = f"Error: {e}"
            
            # Update embed title/color based on tool
            title = "🎶 Music Update"
            color = discord.Color.green()
            if "Error" in res or "Nothing" in res or "Not" in res: color = discord.Color.orange()
            
            if tool_name == "pause_music": title = "⏸️ Paused"
            elif tool_name == "resume_music": title = "⏯️ Resumed"
            elif tool_name == "skip_music": title = "⏭️ Skipped"
            elif tool_name == "stop_music": title = "⏹️ Stopped"
            elif tool_name == "toggle_loop": title = "🔁 Loop"

            embed = self.cog._create_bot_embed(title, res, color, footer="🎵 Music System")
            await interaction.edit_original_response(embed=embed, view=self)

        @discord.ui.button(label="Pause/Resume", style=discord.ButtonStyle.secondary, emoji="⏯️")
        async def pause_resume(self, interaction: discord.Interaction, button: discord.ui.Button):
            # Check if paused or playing to decide which tool to call
            guild = interaction.guild
            voice_client = discord.utils.get(self.cog.bot.voice_clients, guild=guild)
            if voice_client and voice_client.is_paused():
                await self._run_tool(interaction, "resume_music")
            else:
                await self._run_tool(interaction, "pause_music")

        @discord.ui.button(label="Skip", style=discord.ButtonStyle.primary, emoji="⏭️")
        async def skip(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._run_tool(interaction, "skip_music")

        @discord.ui.button(label="Loop", style=discord.ButtonStyle.success, emoji="🔁")
        async def loop_toggle(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._run_tool(interaction, "toggle_loop")

        @discord.ui.button(label="Queue", style=discord.ButtonStyle.secondary, emoji="📜")
        async def queue_list(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._run_tool(interaction, "list_queue")

        @discord.ui.button(label="Stop", style=discord.ButtonStyle.danger, emoji="⏹️")
        async def stop(self, interaction: discord.Interaction, button: discord.ui.Button):
            await self._run_tool(interaction, "stop_music")
            self.stop()

    async def _exec_music_tool(self, interaction: discord.Interaction, tool_name: str, **kwargs):
        user_id = str(interaction.user.id)
        is_dm = interaction.guild is None

        # --- ONE-TIME AGE VERIFICATION FOR DMs (Slash Commands) ---
        if is_dm and user_id not in self.bot.user_age_verified:
            embed = discord.Embed(
                title="🔞 Age Verification Required",
                description="Before using commands in DMs, please verify your age status in our chat history above or send me a DM first.",
                color=discord.Color.gold()
            )
            # Since interaction.response might have been used or deferred elsewhere, we use followup for safety
            if not interaction.response.is_done():
                await interaction.response.send_message(embed=embed, ephemeral=True)
            else:
                await interaction.followup.send(embed=embed, ephemeral=True)
            return

        await interaction.response.defer()
        kwargs['user_id'] = user_id
        
        # Inject NSFW status if calling generate_image
        if tool_name == "generate_image":
            if is_dm:
                kwargs['is_nsfw'] = self.bot.user_age_verified.get(user_id, False)
            else:
                kwargs['is_nsfw'] = getattr(interaction.channel, "nsfw", False)

        tool = next((t for t in self.tools if t.name == tool_name), None)
        if not tool:
            return await interaction.followup.send(embed=self._create_bot_embed("Error", f"Tool '{tool_name}' not found.", discord.Color.red(), footer="🤖 System"))
        
        try:
            # Use ainvoke for all tools as it handles both sync and async internally in LangChain 0.2+
            res = await tool.ainvoke(kwargs)
        except Exception as e:
            res = f"Error: {e}"
        
        color = discord.Color.green()
        if "Error" in res or "Nothing" in res or "Not" in res:
            color = discord.Color.orange()
        
        title = "🎶 Music Update"
        footer = "🎵 Music System"
        image_url = None
        image_file = None
        
        if tool_name == "play_music": title = "▶️ Playback"
        elif tool_name == "stop_music": title = "⏹️ Stopped"
        elif tool_name == "skip_music": title = "⏭️ Skipped"
        elif tool_name == "list_queue": title = "📜 Music Queue"
        elif tool_name == "pause_music": title = "⏸️ Paused"
        elif tool_name == "resume_music": title = "⏯️ Resumed"
        elif tool_name == "set_volume": title = "🔊 Volume"
        elif tool_name == "toggle_loop": title = "🔁 Loop"
        elif tool_name == "set_bass": title = "🎸 Bass"
        elif tool_name == "set_autoplay": title = "📻 Autoplay"
        elif tool_name == "generate_image": 
            title = "🎨 Image Generation"
            footer = "🎨 Image Generation System"
            if "Generated Image: " in res:
                raw_data = res.replace("Generated Image: ", "").strip()
                if raw_data.startswith("data:image"):
                    try:
                        b64_match = re.search(r'data:image/([^;]+);base64,([a-zA-Z0-9+/]+={0,2})', raw_data)
                        if b64_match:
                            ext = b64_match.group(1)
                            b64_data = b64_match.group(2).strip()
                            # Fix padding
                            missing_padding = len(b64_data) % 4
                            if missing_padding:
                                b64_data += '=' * (4 - missing_padding)
                            image_bytes = base64.b64decode(b64_data)
                            image_file = File(io.BytesIO(image_bytes), filename=f"generated_image.{ext}")
                            
                            # --- MEMORY OPTIMIZATION ---
                            # Replace the massive base64 in the result string so it's not stored in history
                            res = f"Generated Image: [File sent as {ext}. Base64 removed for memory.]"
                    except Exception as e:
                        logger.error(f"Error decoding image in slash: {e}")
                else:
                    image_url = raw_data
                res = f"**Prompt:** {kwargs.get('prompt', 'Unknown')}"
                color = discord.Color.random()

        # Only show music controls for actual music tools
        music_tools = [
            "play_music", "pause_music", "resume_music", "skip_music", 
            "stop_music", "toggle_loop", "list_queue", "set_volume", 
            "set_bass", "set_autoplay"
        ]
        view = self.MusicControlView(self, user_id) if tool_name in music_tools else None
        
        embed = self._create_bot_embed(title, res, color, footer=footer)
        if image_url:
            embed.set_image(url=image_url)
        elif image_file:
            embed.set_image(url=f"attachment://{image_file.filename}")
        
        # Build send arguments dynamically to avoid passing None to 'view'
        send_args = {"embed": embed}
        if view: send_args["view"] = view
        if image_file: send_args["file"] = image_file
        
        await interaction.followup.send(**send_args)

    @app_commands.command(name="play", description="Play music.")
    async def play(self, interaction: discord.Interaction, query: str):
        await self._exec_music_tool(interaction, "play_music", query=query)

    @app_commands.command(name="stop", description="Stop music and leave.")
    async def stop(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "stop_music")

    @app_commands.command(name="skip", description="Skip song.")
    async def skip(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "skip_music")

    @app_commands.command(name="queue", description="Show queue.")
    async def queue(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "list_queue")

    @app_commands.command(name="pause", description="Pause music.")
    async def pause(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "pause_music")

    @app_commands.command(name="resume", description="Resume music.")
    async def resume(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "resume_music")

    @app_commands.command(name="volume", description="Set volume (0-100).")
    async def volume(self, interaction: discord.Interaction, level: int):
        await self._exec_music_tool(interaction, "set_volume", volume=level)

    @app_commands.command(name="loop", description="Toggle loop.")
    async def loop(self, interaction: discord.Interaction):
        await self._exec_music_tool(interaction, "toggle_loop")

    @app_commands.command(name="bass", description="Set bass level (0-20).")
    async def bass(self, interaction: discord.Interaction, level: int):
        await self._exec_music_tool(interaction, "set_bass", level=level)

    @app_commands.command(name="autoplay", description="Toggle autoplay.")
    async def autoplay(self, interaction: discord.Interaction, enabled: bool):
        await self._exec_music_tool(interaction, "set_autoplay", enabled=enabled)

    @app_commands.command(name="imagine", description="Generate an image from a prompt.")
    async def imagine(self, interaction: discord.Interaction, prompt: str):
        await self._exec_music_tool(interaction, "generate_image", prompt=prompt)

    @app_commands.command(name="auto_reply", description="Manage auto-reply channels.")
    @app_commands.describe(action="Add, remove, or list channels", channel="The channel mention, ID, or name (defaults to current channel)")
    @app_commands.choices(action=[
        app_commands.Choice(name="Add", value="add"),
        app_commands.Choice(name="Remove", value="remove"),
        app_commands.Choice(name="List", value="list")
    ])
    @app_commands.checks.has_permissions(manage_channels=True)
    async def auto_reply(self, interaction: discord.Interaction, action: app_commands.Choice[str], channel: str = None):
        await interaction.response.defer(ephemeral=True)
        
        if channel:
            # Parse ID from mention <#123> or use raw string if it's just an ID
            match = re.search(r'<#(\d+)>', channel)
            channel_id = None
            if match:
                channel_id = int(match.group(1))
            else:
                try:
                    channel_id = int(channel)
                except ValueError:
                    # Try to find by name
                    clean_name = channel.replace("#", "").strip()
                    found = discord.utils.get(interaction.guild.text_channels, name=clean_name)
                    if found:
                        channel_id = found.id
            
            if not channel_id:
                return await interaction.followup.send("❌ Could not find that channel. Please use a mention (#channel) or a valid ID.", ephemeral=True)
            
            target_channel = interaction.guild.get_channel(channel_id)
        else:
            target_channel = interaction.channel

        # Force check for Text Channel type
        if not isinstance(target_channel, discord.TextChannel):
            return await interaction.followup.send("❌ Auto-reply can only be enabled for standard **Text Channels**. Threads, Forum channels, and Voice channels are not supported.", ephemeral=True)
        
        if action.value == "add":
            self.bot.auto_reply_channels.add(target_channel.id)
            self.bot.save_persistent_config()
            await interaction.followup.send(f"✅ Added {target_channel.mention} to auto-reply channels.", ephemeral=True)
        elif action.value == "remove":
            if target_channel.id in self.bot.auto_reply_channels:
                self.bot.auto_reply_channels.remove(target_channel.id)
                self.bot.save_persistent_config()
                await interaction.followup.send(f"❌ Removed {target_channel.mention} from auto-reply channels.", ephemeral=True)
            else:
                await interaction.followup.send(f"⚠️ {target_channel.mention} is not in the list.", ephemeral=True)
        elif action.value == "list":
            channels = [f"<#{cid}>" for cid in self.bot.auto_reply_channels]
            list_str = "\n".join(channels) if channels else "None"
            await interaction.followup.send(f"### 🤖 Auto-Reply Channels:\n{list_str}", ephemeral=True)

    @app_commands.command(name="vibe_check", description="Server vibe.")
    @app_commands.checks.has_permissions(manage_guild=True)
    async def vibe_check(self, interaction: discord.Interaction):
        await interaction.response.defer()
        history = []
        async for msg in interaction.channel.history(limit=50):
            if not msg.author.bot: history.append(f"{msg.author.name}: {msg.content}")
        if not history: return await interaction.followup.send("No data.")
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Short sentiment analysis."), HumanMessage(content="\n".join(history))])
            await self._send_split_message(interaction, self._get_clean_text(response.content), prefix="### 🌡️ Vibe\n")
        except: await interaction.followup.send("Failed.")

    # --- Help & Prefix Commands ---
    def _get_help_embed(self):
        embed = discord.Embed(
            title="🤖 Bot Help Menu",
            description="Here are the available commands. Most work as both **Slash (/)** and **Prefix (!)**.",
            color=discord.Color.blue()
        )
        
        embed.add_field(
            name="🤖 AI & Chat",
            value=(
                "`/recap` or `!recap` - Summarize last 100 messages.\n"
                "`/ask_memory` - Query bot's memory about you.\n"
                "`/vibe_check` - Analyze server mood (Admin).\n"
                "`/personality` - Change bot persona (Admin)."
            ),
            inline=False
        )
        
        embed.add_field(
            name="🎵 Multimedia & Games",
            value=(
                "`/imagine` - Generate AI images.\n"
                "`/play <query>` - Play music in voice.\n"
                "`/queue`, `/skip`, `/stop` - Music controls.\n"
                "`/start_game <type>` - Start a mini-game."
            ),
            inline=False
        )
        
        embed.add_field(
            name="⚙️ Configuration",
            value=(
                "`/auto_reply <add/remove/list>` - Manage auto-reply channels.\n"
                "`/status` or `!status` - Check bot health.\n"
                "`!sync` - Sync slash commands (Admin)."
            ),
            inline=False
        )
        
        embed.set_footer(text="Tip: Mention me or reply to me to chat anytime!")
        return embed

    @app_commands.command(name="help", description="Show help menu.")
    async def slash_help(self, interaction: discord.Interaction):
        await interaction.response.send_message(embed=self._get_help_embed())

    @commands.command(name="help")
    async def prefix_help(self, ctx: commands.Context):
        await ctx.reply(embed=self._get_help_embed())

    @commands.command(name="ping")
    async def prefix_ping(self, ctx: commands.Context):
        await ctx.reply(f"🏓 {round(self.bot.latency * 1000)}ms")

    @commands.command(name="status")
    async def prefix_status(self, ctx: commands.Context):
        await ctx.reply(f"🤖 Bot ok. Latency: {round(self.bot.latency * 1000)}ms")

    @commands.command(name="recap")
    async def prefix_recap(self, ctx: commands.Context):
        history = []
        async for msg in ctx.channel.history(limit=100):
            if not msg.author.bot: history.append(f"{msg.author.name}: {msg.content}")
        if not history: return await ctx.reply("No data.")
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Summarize in 3 sentences."), HumanMessage(content="\n".join(history[:40]))])
            await self._send_split_message(ctx, self._get_clean_text(response.content), prefix="### 📋 Recap\n")
        except: await ctx.reply("Failed.")

    @commands.command(name="sync")
    async def prefix_sync(self, ctx: commands.Context):
        if ctx.author.id not in ADMIN_IDS:
            return await ctx.reply("❌ No auth.")
        await self.bot.tree.sync()
        await ctx.reply("✅ Slash commands synced.")

    async def check_and_nudge(self, channel: discord.TextChannel):
        history = []
        async for msg in channel.history(limit=10):
            if not msg.author.bot: history.append(f"{msg.author.name}: {msg.content}")
        if len(history) < 5: return
        try:
            response = await self.llm.ainvoke([SystemMessage(content="Toxic? If yes, witty de-escalate. If no, output NO."), HumanMessage(content="\n".join(history))])
            answer = self._get_clean_text(response.content).strip()
            if answer.upper() != "NO": await channel.send(f"👮‍♂️ {answer}")
        except: pass

    @tasks.loop(hours=6)
    async def proactive_starter(self):
        now = discord.utils.utcnow()
        for channel_id in self.bot.auto_reply_channels:
            channel = self.bot.get_channel(channel_id)
            if channel:
                try:
                    last_msg = None
                    async for msg in channel.history(limit=1): last_msg = msg
                    if not last_msg or (now - last_msg.created_at).total_seconds() > (6 * 3600):
                        response = await self.llm.ainvoke([SystemMessage(content=get_personality(self.bot, channel.id)), HumanMessage(content="Short conversation starter.")])
                        await channel.send(self._get_clean_text(response.content))
                except: pass

    @tasks.loop(hours=24)
    async def evolve_persona_task(self):
        logger.info("Evolving persona...")
        recent = vector_store.query("Trends?", n_results=10)
        if not recent: return
        prompt = 'Identify top 3 slang words/topics. JSON: {"slang": [], "trending_topics": []}'
        try:
            resp = await self.llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=recent)])
            data = json.loads(self._get_clean_text(resp.content).replace("```json", "").replace("```", ""))
            self.bot.server_culture.update(data)
            self.bot.save_persistent_config()
        except: pass

    @commands.Cog.listener()
    async def on_ready(self):
        if not self.evolve_persona_task.is_running(): self.evolve_persona_task.start()

async def setup(bot):
    await bot.add_cog(AIAgent(bot))
