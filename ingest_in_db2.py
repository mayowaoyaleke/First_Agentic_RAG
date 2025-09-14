# import basics
import os
from dotenv import load_dotenv
import requests
from urllib.parse import urljoin, urlparse
from typing import List

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.document_transformers import Html2TextTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# import supabase
from supabase.client import Client, create_client

# import for web scraping
import asyncio
from bs4 import BeautifulSoup

# load environment variables
load_dotenv()  

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_website_content(urls: List[str]) -> List[Document]:
    """Load content from a list of URLs"""
    documents = []
    
    # Method 1: Using WebBaseLoader (simpler)
    loader = WebBaseLoader(urls)
    web_docs = loader.load()
    
    # Add metadata to distinguish web content
    for doc in web_docs:
        doc.metadata['source_type'] = 'website'
        doc.metadata['url'] = doc.metadata.get('source', '')
        documents.extend([doc])
    
    return documents

def load_website_content_async(urls: List[str]) -> List[Document]:
    """Load content from URLs asynchronously (faster for multiple URLs)"""
    # Load HTML content
    loader = AsyncHtmlLoader(urls)
    html_docs = loader.load()
    
    # Transform HTML to text
    html2text = Html2TextTransformer()
    docs = html2text.transform_documents(html_docs)
    
    # Add metadata
    for doc in docs:
        doc.metadata['source_type'] = 'website'
        doc.metadata['url'] = doc.metadata.get('source', '')
    
    return docs

def scrape_sitemap(sitemap_url: str) -> List[str]:
    """Extract URLs from a sitemap.xml"""
    try:
        response = requests.get(sitemap_url)
        soup = BeautifulSoup(response.content, 'xml')
        urls = [loc.text for loc in soup.find_all('loc')]
        return urls
    except Exception as e:
        print(f"Error scraping sitemap: {e}")
        return []

def crawl_website(base_url: str, max_pages: int = 10) -> List[str]:
    """Simple website crawler to discover pages"""
    visited = set()
    to_visit = [base_url]
    urls = []
    
    while to_visit and len(urls) < max_pages:
        current_url = to_visit.pop(0)
        if current_url in visited:
            continue
            
        visited.add(current_url)
        urls.append(current_url)
        
        try:
            response = requests.get(current_url, timeout=10)
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find links to other pages on the same domain
            for link in soup.find_all('a', href=True):
                href = link['href']
                full_url = urljoin(current_url, href)
                domain = urlparse(base_url).netloc
                
                if urlparse(full_url).netloc == domain and full_url not in visited:
                    to_visit.append(full_url)
                    
        except Exception as e:
            print(f"Error crawling {current_url}: {e}")
            continue
    
    return urls

# Configuration: Define your website URLs
website_urls = [
    "https://example.com/page1",
    "https://example.com/page2",
    "https://example.com/about",
]

# Alternative: Crawl a website automatically
# crawled_urls = crawl_website("https://example.com", max_pages=20)

# Alternative: Load from sitemap
# sitemap_urls = scrape_sitemap("https://example.com/sitemap.xml")

# Load PDF documents (your existing code)
pdf_loader = PyPDFDirectoryLoader("documents")
pdf_documents = pdf_loader.load()

# Add metadata to PDF documents
for doc in pdf_documents:
    doc.metadata['source_type'] = 'pdf'

# Load website content
print("Loading website content...")
web_documents = load_website_content(website_urls)
# For better performance with many URLs, use:
# web_documents = load_website_content_async(website_urls)

# Combine all documents
all_documents = pdf_documents + web_documents

print(f"Loaded {len(pdf_documents)} PDF documents and {len(web_documents)} web pages")

# Split documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=100,
    # Add separators that work well for web content
    separators=["\n\n", "\n", " ", ""]
)

docs = text_splitter.split_documents(all_documents)

print(f"Created {len(docs)} document chunks")

# Store chunks in vector store
vector_store = SupabaseVectorStore.from_documents(
    docs,
    embeddings,
    client=supabase,
    table_name="documents",
    query_name="match_documents",
    chunk_size=1000,
)

print("Documents successfully stored in vector database!")

# Optional: Create a function to query and show source information
def query_with_sources(query: str, k: int = 5):
    """Query the vector store and return results with source information"""
    results = vector_store.similarity_search_with_score(query, k=k)
    
    for i, (doc, score) in enumerate(results, 1):
        print(f"\n--- Result {i} (Score: {score:.3f}) ---")
        print(f"Content: {doc.page_content[:200]}...")
        print(f"Source Type: {doc.metadata.get('source_type', 'unknown')}")
        
        if doc.metadata.get('source_type') == 'website':
            print(f"URL: {doc.metadata.get('url', 'N/A')}")
        else:
            print(f"File: {doc.metadata.get('source', 'N/A')}")

# Example usage:
# query_with_sources("What is the company's mission?")