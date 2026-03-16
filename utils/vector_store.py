import os
from typing import List, Dict, Optional
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

class MessageVectorStore:
    def __init__(self, collection_name="discord_messages_v3", persist_directory="./chroma_db"):
        self.provider = os.getenv("AI_PROVIDER", "openai").lower()
        self.embeddings = self._init_embeddings()
        
        self.vector_store = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=persist_directory
        )

    def _init_embeddings(self):
        """
        Initialize embeddings. Prioritize Google (Free) or Local to avoid OpenAI costs.
        """
        if self.provider == "google" or os.getenv("GOOGLE_API_KEY"):
            # Google AI Studio Free Tier - 1500 RPM for embeddings
            return GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
        elif self.provider == "local":
            # 100% Free - Runs on your CPU
            return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Fallback to OpenAI only if explicitly requested
        return OpenAIEmbeddings(
            model="text-embedding-3-small",
            api_key=os.getenv("OPENAI_API_KEY")
        )

    async def add_messages(self, messages: List[Dict]):
        documents = [
            Document(
                page_content=f"{msg['author']}: {msg['content']}",
                metadata={
                    "id": msg["id"],
                    "timestamp": msg["timestamp"],
                    "author": msg["author"],
                    "author_id": msg["author_id"]
                }
            ) for msg in messages
        ]
        self.vector_store.add_documents(documents)

    def query(self, query_text: str, n_results=5, filter_dict: Optional[Dict] = None) -> str:
        """
        Search for similar documents. Optional filter to isolate user history.
        """
        results = self.vector_store.similarity_search(query_text, k=n_results, filter=filter_dict)
        return "\n".join([doc.page_content for doc in results])

# Global instance
vector_store = MessageVectorStore()
