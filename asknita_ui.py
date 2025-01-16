from __future__ import annotations
import os
import asyncio
from typing import List, Dict, Any
from datetime import datetime

import streamlit as st
import json
from supabase import create_client, Client
from mistralai.client import MistralClient
from groq import AsyncGroq
import ollama
from openai import AsyncOpenAI as OpenRouter  # OpenRouter uses OpenAI's interface

from dotenv import load_dotenv
load_dotenv()

# Initialize API clients
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "PNVuwy0j47BptZ63KL9jqMysILLD2ZhN"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "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"))
openrouter_client = OpenRouter(
    api_key=os.getenv("OPENROUTER_API_KEY", "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"),
    base_url="https://openrouter.ai/api/v1"
)
ollama_client = ollama.AsyncClient(host='http://localhost:11434')

# Initialize Supabase client
supabase_url = os.getenv("SUPABASE_URL", "https://xefcmcpaddxowqdulxah.supabase.co")
supabase_key = os.getenv("SUPABASE_SERVICE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InhlZmNtY3BhZGR4b3dxZHVseGFoIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTczNzAwNTU2MiwiZXhwIjoyMDUyNTgxNTYyfQ.Zq8Hydi4K_KsFQ1OrYUBiEG8oMUxvbQPKI761OSuzGM")
supabase: Client = create_client(supabase_url, supabase_key)

# Cache for API providers to handle rate limits
class APIProvider:
    def __init__(self, name: str, client: Any, is_async: bool = False):
        self.name = name
        self.client = client
        self.is_async = is_async
        self.error_count = 0
        self.last_error_time = None

    def should_retry(self) -> bool:
        if self.last_error_time is None:
            return True
        if (datetime.now() - self.last_error_time).total_seconds() > 300:
            self.error_count = 0
            return True
        return self.error_count < 3

api_providers = {
    'chat': [
        APIProvider('mistral', mistral_client, False),
        APIProvider('groq', groq_client, True),
        APIProvider('openrouter', openrouter_client, True),
        APIProvider('ollama', ollama_client, True)
    ],
    'embedding': [
        APIProvider('mistral', mistral_client, False),
        APIProvider('groq', groq_client, True),
        APIProvider('openrouter', openrouter_client, True),
        APIProvider('ollama', ollama_client, True)
    ]
}

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from available API providers."""
    # Truncate text to avoid rate limits
    max_tokens = 1000  # Conservative limit
    text = ' '.join(text.split()[:max_tokens])
    
    for provider in api_providers['embedding']:
        if not provider.should_retry():
            continue

        try:
            if provider.name == 'mistral':
                response = provider.client.embeddings(
                    model="mistral-embed",
                    input=text
                )
                # Mistral API returns a dict directly
                return response.data[0]['embedding']
            
            elif provider.name == 'groq':
                response = await provider.client.embeddings.create(
                    model="mixtral-8x7b-32768",
                    input=text
                )
                # Groq API returns an object with embedding attribute
                return response.data[0].embedding
                
            elif provider.name == 'openrouter':
                response = await provider.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=text
                )
                return response.data[0].embedding
                
            elif provider.name == 'ollama':
                response = await provider.client.embeddings(
                    model="mistral",
                    prompt=text
                )
                return response['embedding']

        except Exception as e:
            print(f"Error with {provider.name} embeddings API: {e}")
            provider.error_count += 1
            provider.last_error_time = datetime.now()
            continue

    return [0.0] * 1024  # Using 1024 dimensions as default

async def get_relevant_context(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Get relevant context from the database using vector similarity search."""
    try:
        # Get embedding for the query
        query_embedding = await get_embedding(query)
        
        # Search for similar content in the database
        response = supabase.rpc(
            'match_nita_pages',
            {
                'query_embedding': query_embedding,
                'match_count': max_results,
                'filter': {}
            }
        ).execute()
        
        return response.data
    except Exception as e:
        print(f"Error getting relevant context: {e}")
        return []

async def get_ai_response(query: str, context: List[Dict[str, Any]]) -> str:
    """Get AI response using available providers."""
    system_prompt = """You are ASKNITA, an AI assistant for NIT Agartala. Your purpose is to help users with questions about the institute.
    Use the provided context to answer questions accurately and helpfully. If you're not sure about something, say so.
    Always maintain a professional and friendly tone. Format your responses in markdown for better readability."""
    
    context_text = "\n\n".join([
        f"From {doc['url']}:\nTitle: {doc['title']}\nContent: {doc['content']}"
        for doc in context
    ])
    
    for provider in api_providers['chat']:
        if not provider.should_retry():
            continue

        try:
            if provider.name == 'mistral':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ]
                response = provider.client.chat(
                    model="mistral-medium",
                    messages=messages
                )
                return response.choices[0].message.content
            
            elif provider.name == 'groq':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ]
                response = await provider.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=messages
                )
                return response.choices[0].message.content
                
            elif provider.name == 'openrouter':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ]
                response = await provider.client.chat.completions.create(
                    model="text-davinci-003",
                    messages=messages
                )
                return response.choices[0].message.content
                
            elif provider.name == 'ollama':
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
                ]
                response = await provider.client.chat(
                    model="mistral",
                    messages=messages
                )
                return response['message']

        except Exception as e:
            print(f"Error with {provider.name} API: {e}")
            provider.error_count += 1
            provider.last_error_time = datetime.now()
            continue

    return "I apologize, but I'm having trouble accessing the AI services at the moment. Please try again in a few minutes."

def initialize_session_state():
    """Initialize session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'show_sources' not in st.session_state:
        st.session_state.show_sources = False

async def main():
    st.set_page_config(
        page_title="ASKNITA - NIT Agartala AI Assistant",
        page_icon="🎓",
        layout="wide"
    )

    st.title("🎓 ASKNITA - Your NIT Agartala AI Assistant")
    
    # Sidebar for settings
    with st.sidebar:
        st.header("Settings")
        st.session_state.show_sources = st.checkbox("Show source documents", value=False)
        
        st.header("About")
        st.markdown("""
        ASKNITA is your AI-powered assistant for all things NIT Agartala. Ask questions about:
        - 📚 Academic programs and courses
        - 📝 Admission procedures
        - 🏛️ Campus facilities
        - 🔬 Research activities
        - 👨‍🏫 Faculty information
        - 📅 Events and notices
        - And much more!
        """)

    initialize_session_state()

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant" and "sources" in message and st.session_state.show_sources:
                st.markdown("---\n**Sources:**")
                for source in message["sources"]:
                    st.markdown(f"- [{source['title']}]({source['url']})")

    # Chat input
    if prompt := st.chat_input("Ask me anything about NIT Agartala..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get AI response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("🤔 Let me search through the NITA documentation...")
            
            # Get relevant context
            context = await get_relevant_context(prompt)
            
            # Get AI response
            response = await get_ai_response(prompt, context)
            
            # Update message placeholder with response
            message_placeholder.markdown(response)
            
            # Add response to chat history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response,
                "sources": [{
                    "title": doc["title"],
                    "url": doc["url"]
                } for doc in context]
            })

if __name__ == "__main__":
    asyncio.run(main())
