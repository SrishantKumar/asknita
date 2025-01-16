import os
import sys
import json
import asyncio
import requests
from xml.etree import ElementTree
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timezone
from urllib.parse import urlparse
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from mistralai.client import MistralClient
from groq import AsyncGroq

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode
from supabase import create_client, Client

load_dotenv()

# Initialize API clients
mistral_client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY", "PNVuwy0j47BptZ63KL9jqMysILLD2ZhN"))
groq_client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY", "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"))
supabase: Client = create_client(
    os.getenv("SUPABASE_URL"),
    os.getenv("SUPABASE_SERVICE_KEY")
)

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
        # Reset error count after 5 minutes
        if (datetime.now() - self.last_error_time).total_seconds() > 300:
            self.error_count = 0
            return True
        return self.error_count < 3

api_providers = {
    'chat': [
        APIProvider('mistral', mistral_client, False),
        APIProvider('groq', groq_client, True)
    ],
    'embedding': [
        APIProvider('mistral', mistral_client, False),
        APIProvider('groq', groq_client, True)
    ]
}

@dataclass
class ProcessedChunk:
    url: str
    chunk_number: int
    title: str
    summary: str
    content: str
    metadata: Dict[str, Any]
    embedding: List[float]

def chunk_text(text: str, chunk_size: int = 5000) -> List[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind('```')
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block

        # If no code block, try to break at a paragraph
        elif '\n\n' in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind('\n\n')
            if last_break > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_break

        # If no paragraph break, try to break at a sentence
        elif '. ' in chunk:
            # Find the last sentence break
            last_period = chunk.rfind('. ')
            if last_period > chunk_size * 0.3:  # Only break if we're past 30% of chunk_size
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = max(start + 1, end)

    return chunks

async def get_title_and_summary(chunk: str, url: str) -> Dict[str, str]:
    """Extract title and summary using available API providers."""
    system_prompt = """You are an AI that extracts titles and summaries from documentation chunks.
    Return a JSON object with 'title' and 'summary' keys.
    For the title: If this seems like the start of a document, extract its title. If it's a middle chunk, derive a descriptive title.
    For the summary: Create a concise summary of the main points in this chunk.
    Keep both title and summary concise but informative."""

    for provider in api_providers['chat']:
        if not provider.should_retry():
            continue

        try:
            if provider.name == 'mistral':
                response = provider.client.chat(
                    model="mistral-medium",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                    ]
                )
                return json.loads(response.choices[0].message.content)
            
            elif provider.name == 'groq':
                response = await provider.client.chat.completions.create(
                    model="mixtral-8x7b-32768",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"URL: {url}\n\nContent:\n{chunk[:1000]}..."}
                    ],
                    response_format={ "type": "json_object" }
                )
                return json.loads(response.choices[0].message.content)

        except Exception as e:
            print(f"Error with {provider.name} API: {e}")
            provider.error_count += 1
            provider.last_error_time = datetime.now()
            continue

    # If all providers fail, return a default response
    return {
        "title": f"Page from {url.split('/')[-1]}",
        "summary": "Content from NITA website"
    }

async def get_embedding(text: str) -> List[float]:
    """Get embedding vector from available API providers."""
    for provider in api_providers['embedding']:
        if not provider.should_retry():
            continue

        try:
            if provider.name == 'mistral':
                response = provider.client.embeddings(
                    model="mistral-embed",
                    input=text
                )
                return response.data[0].embedding
            
            elif provider.name == 'groq':
                response = await provider.client.embeddings.create(
                    model="mixtral-8x7b-32768",
                    input=text
                )
                return response.data[0].embedding

        except Exception as e:
            print(f"Error with {provider.name} embeddings API: {e}")
            provider.error_count += 1
            provider.last_error_time = datetime.now()
            continue

    # If all providers fail, return a zero vector
    return [0.0] * 1024  # Using 1024 dimensions as default

async def process_chunk(chunk: str, chunk_number: int, url: str) -> ProcessedChunk:
    """Process a single chunk of text."""
    # Get title and summary
    extracted = await get_title_and_summary(chunk, url)
    
    # Get embedding
    embedding = await get_embedding(chunk)
    
    # Create metadata
    metadata = {
        "source": "nita_docs",
        "chunk_size": len(chunk),
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "url_path": urlparse(url).path
    }
    
    return ProcessedChunk(
        url=url,
        chunk_number=chunk_number,
        title=extracted['title'],
        summary=extracted['summary'],
        content=chunk,  # Store the original chunk content
        metadata=metadata,
        embedding=embedding
    )

async def insert_chunk(chunk: ProcessedChunk):
    """Insert a processed chunk into Supabase."""
    try:
        data = {
            'url': chunk.url,
            'chunk_number': chunk.chunk_number,
            'title': chunk.title,
            'summary': chunk.summary,
            'content': chunk.content,
            'metadata': chunk.metadata,
            'embedding': chunk.embedding
        }
        
        response = supabase.table('nita_pages').insert(data).execute()
        print(f"Successfully inserted chunk {chunk.chunk_number} from {chunk.url}")
        return response
    except Exception as e:
        print(f"Error inserting chunk: {e}")
        return None

async def process_and_store_document(url: str, markdown: str):
    """Process a document and store its chunks in parallel."""
    # Split into chunks
    chunks = chunk_text(markdown)
    
    # Process chunks in parallel
    tasks = [
        process_chunk(chunk, i, url) 
        for i, chunk in enumerate(chunks)
    ]
    processed_chunks = await asyncio.gather(*tasks)
    
    # Store chunks in parallel
    insert_tasks = [
        insert_chunk(chunk) 
        for chunk in processed_chunks
    ]
    await asyncio.gather(*insert_tasks)

async def crawl_parallel(urls: List[str], max_concurrent: int = 5):
    """Crawl multiple URLs in parallel with a concurrency limit."""
    # Reduce max concurrent to avoid rate limits
    max_concurrent = 3  
    browser_config = BrowserConfig(
        headless=True,
        verbose=False,
        extra_args=["--disable-gpu", "--disable-dev-shm-usage", "--no-sandbox"],
    )
    crawl_config = CrawlerRunConfig(cache_mode=CacheMode.BYPASS)

    # Create the crawler instance
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        # Create a semaphore to limit concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_url(url: str):
            async with semaphore:
                result = await crawler.arun(
                    url=url,
                    config=crawl_config,
                    session_id="session1"
                )
                if result.success:
                    print(f"Successfully crawled: {url}")
                    await process_and_store_document(url, result.markdown_v2.raw_markdown)
                else:
                    print(f"Failed: {url} - Error: {result.error_message}")
        
        # Process all URLs in parallel with limited concurrency
        await asyncio.gather(*[process_url(url) for url in urls])
    finally:
        await crawler.close()

async def get_nita_urls() -> List[str]:
    """Get URLs from NITA website by crawling the main page."""
    base_url = "https://www.nita.ac.in"
    urls = set()
    
    try:
        # Start with the main page
        response = requests.get(base_url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find all links on the page
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Handle relative URLs
            if href.startswith('/'):
                href = base_url + href
            elif not href.startswith('http'):
                href = base_url + '/' + href
            
            # Only include NITA domain URLs
            if href.startswith(base_url):
                urls.add(href)
        
        print(f"Found {len(urls)} URLs to crawl")
        return list(urls)
    except Exception as e:
        print(f"Error crawling website: {e}")
        return []

async def main():
    """Main entry point for the crawler."""
    print("Starting NITA documentation crawler...")
    
    # Get URLs from website
    urls = await get_nita_urls()
    
    if not urls:
        print("No URLs found. Exiting.")
        return
    
    # Crawl URLs in parallel
    await crawl_parallel(urls)
    print("Crawling complete!")

if __name__ == "__main__":
    asyncio.run(main())
