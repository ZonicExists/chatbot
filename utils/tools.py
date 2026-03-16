import os
import discord
import feedparser
import json
import yt_dlp
import asyncio
import logging
import aiohttp
from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from typing import Optional, List, Annotated
from utils.vector_store import vector_store

# Configure Logging for tools
logger = logging.getLogger("Tools")

# ytdl options
ytdl_format_options = {
    'format': 'bestaudio/best',
    'outtmpl': '%(extractor)s-%(id)s-%(title)s.%(ext)s',
    'restrictfilenames': True,
    'noplaylist': True,
    'nocheckcertificate': True,
    'ignoreerrors': False,
    'logtostderr': False,
    'quiet': True,
    'no_warnings': True,
    'default_search': 'auto',
    'source_address': '0.0.0.0',
}

ffmpeg_options = {
    'options': '-vn',
    'before_options': '-reconnect 1 -reconnect_streamed 1 -reconnect_delay_max 5'
}

ytdl = yt_dlp.YoutubeDL(ytdl_format_options)

class YTDLSource(discord.PCMVolumeTransformer):
    def __init__(self, source, *, data, volume=0.5):
        super().__init__(source, volume)
        self.data = data
        self.title = data.get('title')
        self.url = data.get('url')

    @classmethod
    async def from_url(cls, url, *, loop=None, stream=True):
        loop = loop or asyncio.get_event_loop()
        data = await loop.run_in_executor(None, lambda: ytdl.extract_info(url, download=not stream))
        if 'entries' in data:
            data = data['entries'][0]
        filename = data['url'] if stream else ytdl.prepare_filename(data)
        return cls(discord.FFmpegPCMAudio(filename, **ffmpeg_options), data=data)

def get_tools(bot):
    if not hasattr(bot, 'music_state'):
        bot.music_state = {} # guild_id -> {'loop': False, 'bass': 0, 'queue': [], 'disconnect_task': None, 'autoplay': False, 'last_song': None, 'requester_id': None}

    def get_guild_state(guild_id):
        if guild_id not in bot.music_state:
            bot.music_state[guild_id] = {'loop': False, 'bass': 0, 'queue': [], 'disconnect_task': None, 'autoplay': False, 'last_song': None, 'requester_id': None}
        return bot.music_state[guild_id]

    async def play_next(guild_id, voice_client):
        state = get_guild_state(guild_id)
        
        # Check if requester is still in VC
        if state['requester_id']:
            guild = bot.get_guild(guild_id)
            requester = guild.get_member(int(state['requester_id']))
            if not requester or not requester.voice or requester.voice.channel != voice_client.channel:
                print(f"Requester {state['requester_id']} left VC. Stopping.")
                state['queue'] = []
                state['last_song'] = None
                state['requester_id'] = None
                await voice_client.disconnect()
                return

        # Cancel any existing disconnect timer
        if state['disconnect_task']:
            state['disconnect_task'].cancel()
            state['disconnect_task'] = None

        next_query = None
        if state['queue']:
            next_query = state['queue'].pop(0)
        elif state['autoplay'] and state['last_song']:
            # Fetch recommended song based on last_song
            try:
                search_query = f"ytsearch5:related songs to {state['last_song']}"
                loop = bot.loop or asyncio.get_event_loop()
                info = await loop.run_in_executor(None, lambda: ytdl.extract_info(search_query, download=False))
                if 'entries' in info and len(info['entries']) > 1:
                    # Pick the second or third result to avoid playing the same song again
                    # Usually the first result is the song itself
                    idx = 1 if len(info['entries']) > 1 else 0
                    next_query = info['entries'][idx]['webpage_url']
                    print(f"Autoplay recommending: {info['entries'][idx]['title']}")
            except Exception as e:
                print(f"Error fetching autoplay recommendation: {e}")

        if next_query:
            try:
                # Apply bass filter if set
                current_ffmpeg_options = ffmpeg_options.copy()
                if state['bass'] > 0:
                    current_ffmpeg_options['options'] += f' -af "bass=g={state["bass"]}"'

                player = await YTDLSource.from_url(next_query, loop=bot.loop, stream=True)
                state['last_song'] = player.title
                
                def after_playing(error):
                    if error: print(f"Player error: {error}")
                    if state['loop'] and not error:
                        state['queue'].insert(0, next_query)
                    
                    # Schedule next song or disconnect timer
                    asyncio.run_coroutine_threadsafe(play_next(guild_id, voice_client), bot.loop)

                voice_client.play(player, after=after_playing)
            except Exception as e:
                print(f"Error playing next: {e}")
                asyncio.run_coroutine_threadsafe(play_next(guild_id, voice_client), bot.loop)
        else:
            # No more songs, start 30s disconnect timer
            async def _wait_and_disconnect():
                await asyncio.sleep(30)
                if voice_client.is_connected() and not voice_client.is_playing() and not state['queue']:
                    await voice_client.disconnect()
                    print(f"Disconnected from {guild_id} due to inactivity.")
            
            state['disconnect_task'] = bot.loop.create_task(_wait_and_disconnect())

    # Web Search Tool
    web_search_tool = TavilySearch(max_results=3, tavily_api_key=os.getenv("TAVILY_API_KEY"))

    @tool
    def play_music(
        query: Annotated[str, "The name or URL of the song to play."],
        user_id: Annotated[str, "The Discord ID of the user."]
    ) -> str:
        """Plays music. If music is already playing, it adds to the queue."""
        user_id_int = int(user_id)
        member = None
        guild = None
        
        # Search for the member in any mutual guild with a voice channel
        for g in bot.guilds:
            m = g.get_member(user_id_int)
            if m and m.voice and m.voice.channel:
                member, guild = m, g; break
        
        if not member: return "Error: Music only works in servers. Join a voice channel in a server I'm in!"
        
        state = get_guild_state(guild.id)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)

        async def _play_logic():
            nonlocal voice_client
            if not voice_client:
                voice_client = await member.voice.channel.connect(self_deaf=True)
            elif voice_client.channel != member.voice.channel:
                await voice_client.move_to(member.voice.channel)

            state['requester_id'] = user_id # Store who started/added to this session

            if voice_client.is_playing() or voice_client.is_paused():
                state['queue'].append(query)
                return f"Added to queue: {query}"
            else:
                state['queue'].append(query)
                await play_next(guild.id, voice_client)
                return f"Started playing: {query}"

        return asyncio.run_coroutine_threadsafe(_play_logic(), bot.loop).result()

    @tool
    def set_autoplay(enabled: bool, user_id: str) -> str:
        """Enables or disables autoplay of related songs when the queue is empty."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        if not guild: return "Error: Guild not found."
        state = get_guild_state(guild.id)
        state['autoplay'] = enabled
        return f"Autoplay: {'ENABLED' if state['autoplay'] else 'DISABLED'}."

    @tool
    def list_queue(user_id: Annotated[str, "The Discord ID of the user."]) -> str:
        """Lists the current music queue."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        if not guild: return "Error: Guild not found."
        state = get_guild_state(guild.id)
        if not state['queue']: return "The queue is empty."
        return "Music Queue:\n" + "\n".join([f"{i+1}. {q}" for i, q in enumerate(state['queue'])])

    @tool
    def skip_music(user_id: Annotated[str, "The Discord ID of the user."]) -> str:
        """Skips the current song."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client and (voice_client.is_playing() or voice_client.is_paused()):
            # Temporarily disable loop to skip
            state = get_guild_state(guild.id)
            was_looping = state['loop']
            state['loop'] = False
            voice_client.stop()
            state['loop'] = was_looping
            return "Skipped."
        return "Nothing playing."

    @tool
    def stop_music(user_id: Annotated[str, "The Discord ID of the user."]) -> str:
        """Stops music, clears queue, and leaves."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client:
            state = get_guild_state(guild.id)
            state['queue'] = []
            state['loop'] = False
            state['last_song'] = None
            if state['disconnect_task']: state['disconnect_task'].cancel()
            asyncio.run_coroutine_threadsafe(voice_client.disconnect(), bot.loop)
            return "Stopped and left."
        return "Not in VC."

    @tool
    def pause_music(user_id: Annotated[str, "The Discord ID of the user."]) -> str:
        """Pauses playback."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client and voice_client.is_playing():
            voice_client.pause()
            return "Paused."
        return "Not playing."

    @tool
    def resume_music(user_id: Annotated[str, "The Discord ID of the user."]) -> str:
        """Resumes playback."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client and voice_client.is_paused():
            voice_client.resume()
            return "Resumed."
        return "Not paused."

    @tool
    def set_volume(volume: int, user_id: str) -> str:
        """Sets volume 0-100."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        voice_client = discord.utils.get(bot.voice_clients, guild=guild)
        if voice_client and voice_client.source:
            voice_client.source.volume = volume / 100
            return f"Volume: {volume}%."
        return "Not playing."

    @tool
    def toggle_loop(user_id: str) -> str:
        """Toggles loop."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        state = get_guild_state(guild.id)
        state['loop'] = not state['loop']
        return f"Loop: {'ON' if state['loop'] else 'OFF'}."

    @tool
    def set_bass(level: int, user_id: str) -> str:
        """Sets bass 0-20."""
        user_id_int = int(user_id)
        guild = next((g for g in bot.guilds if g.get_member(user_id_int)), None)
        state = get_guild_state(guild.id)
        state['bass'] = level
        return f"Bass: {level}. (Next song)"

    # --- Other Tools ---
    @tool
    def query_long_term_memory(query: str, author_id: str) -> str:
        """Search past chat."""
        res = vector_store.query(query, n_results=2, filter_dict={"author_id": author_id})
        return f"Context:\n{res}" if res else "No memories."

    @tool
    def get_server_stats(guild_id: Annotated[str, "The Discord Guild/Server ID to check."]) -> str:
        """Retrieves server information like member count and name. Only works for servers the bot is in."""
        try:
            if not guild_id or guild_id == "dm" or guild_id == "0":
                return "Error: This tool only works within a Discord server, not in DMs. Please provide a valid Server ID if you want stats for a specific server I'm in."
            
            guild_id_int = int(guild_id)
            guild = bot.get_guild(guild_id_int)
            if not guild: return f"Error: I couldn't find a server with ID {guild_id}. I must be a member of the server to see its stats."
            return f"Server Stats for {guild.name}:\n- Members: {guild.member_count}\n- Created: {guild.created_at.strftime('%Y-%m-%d')}\n- Description: {guild.description or 'None'}"
        except ValueError:
            return "Error: Invalid Server ID format. Please provide a numeric ID."
        except Exception as e:
            return f"Error getting stats: {e}"

    @tool
    async def generate_image(
        prompt: Annotated[str, "The descriptive prompt for the image to generate."],
        is_nsfw: Annotated[bool, "Whether to allow NSFW content. ONLY set to True if the channel is age-restricted."] = False
    ) -> str:
        """Generates an image based on a text prompt via Pollinations.ai."""
        image_model = os.getenv("IMAGE_MODEL", "pollinations/flux")
        model_name = image_model.split("/")[-1] if "/" in image_model else "flux"
        
        final_prompt = prompt
        if is_nsfw:
            final_prompt += " (Allow adult/NSFW content)"

        logger.info(f"Generating image via Pollinations ({model_name}). NSFW: {is_nsfw}")
        
        import urllib.parse
        encoded_prompt = urllib.parse.quote(final_prompt)
        api_key = os.getenv("POLLINATIONS_API_KEY")
        
        params = {
            "model": model_name,
            "nologo": "true",
            "private": "true",
            "enhance": "false"
        }
        
        query_string = urllib.parse.urlencode(params)
        base_url = os.getenv("POLLINATIONS_IMAGE_URL", "https://gen.pollinations.ai/image/")
        url = f"{base_url}{encoded_prompt}?{query_string}"
        
        logger.info(f"Pollinations Image Prompt: {final_prompt}")
        
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as resp:
                    if resp.status != 200:
                        err_text = await resp.text()
                        logger.error(f"Pollinations Error {resp.status}: {err_text}")
                        return f"Error: Pollinations returned {resp.status}."
                    
                    import base64
                    image_data = await resp.read()
                    b64_image = base64.b64encode(image_data).decode("utf-8")
                    mime_type = resp.headers.get("Content-Type", "image/png")
                    return f"Generated Image: data:{mime_type};base64,{b64_image}"
        except Exception as e:
            logger.error(f"Exception in generate_image: {e}")
            return f"Error: {e}"

    return [
        web_search_tool, query_long_term_memory, get_server_stats,
        play_music, stop_music, skip_music, list_queue,
        pause_music, resume_music, set_volume, toggle_loop, set_bass,
        set_autoplay, generate_image
    ]
