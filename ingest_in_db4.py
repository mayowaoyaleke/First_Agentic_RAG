# import basics
import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from typing import List

# import langchain
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import SupabaseVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document

# import supabase
from supabase.client import Client, create_client

# load environment variables
load_dotenv()  

# Set user agent to avoid warnings
os.environ['USER_AGENT'] = 'MyRAGSystem/1.0'

# Configure detailed logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'rag_ingestion_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def log_document_details(documents: List[Document], source_type: str):
    """Log detailed information about loaded documents"""
    logger.info(f"\n{'='*60}")
    logger.info(f"{source_type.upper()} DOCUMENT DETAILS")
    logger.info(f"{'='*60}")
    
    for i, doc in enumerate(documents, 1):
        logger.info(f"Document {i}:")
        logger.info(f"  📄 Source: {doc.metadata.get('source', 'Unknown')}")
        logger.info(f"  📏 Content length: {len(doc.page_content)} characters")
        logger.info(f"  🏷️  Metadata keys: {list(doc.metadata.keys())}")
        
        # Show first 200 characters of content
        preview = doc.page_content[:200].replace('\n', ' ').strip()
        logger.info(f"  👀 Content preview: {preview}...")
        
        # Log specific metadata based on source type
        if source_type == 'pdf':
            page = doc.metadata.get('page', 'Unknown')
            logger.info(f"  📖 Page: {page}")
        elif source_type == 'website':
            url = doc.metadata.get('url', 'Unknown')
            title = doc.metadata.get('title', 'No title')
            logger.info(f"  🌐 URL: {url}")
            logger.info(f"  📰 Title: {title}")
        
        logger.info("  " + "-"*50)

def load_pdfs():
    """Load PDF documents from the documents folder"""
    logger.info("🔄 Starting PDF document loading...")
    
    # Check if documents directory exists
    docs_dir = "documents"
    if not os.path.exists(docs_dir):
        logger.warning(f"❌ Documents directory '{docs_dir}' does not exist")
        return []
    
    # List PDF files in directory
    pdf_files = [f for f in os.listdir(docs_dir) if f.lower().endswith('.pdf')]
    logger.info(f"📁 Found {len(pdf_files)} PDF files in directory:")
    for pdf_file in pdf_files:
        file_size = os.path.getsize(os.path.join(docs_dir, pdf_file))
        logger.info(f"  - {pdf_file} ({file_size:,} bytes)")
    
    if not pdf_files:
        logger.warning(f"⚠️  No PDF files found in '{docs_dir}' directory")
        return []
    
    try:
        loader = PyPDFDirectoryLoader(docs_dir)
        documents = loader.load()
        
        # Add source type metadata
        for doc in documents:
            doc.metadata['source_type'] = 'pdf'
            doc.metadata['loaded_at'] = datetime.now().isoformat()
            
        logger.info(f"✅ Successfully loaded {len(documents)} PDF pages")
        
        # Log detailed document information
        log_document_details(documents, 'pdf')
        
        return documents
        
    except Exception as e:
        logger.error(f"❌ Error loading PDFs: {e}")
        return []

def load_websites(urls: List[str]):
    """Load content from website URLs"""
    if not urls:
        logger.info("⚠️  No website URLs provided, skipping web loading...")
        return []
        
    logger.info(f"🔄 Starting website loading for {len(urls)} URLs...")
    web_documents = []
    
    for i, url in enumerate(urls, 1):
        logger.info(f"📡 Loading website {i}/{len(urls)}: {url}")
        
        try:
            loader = WebBaseLoader([url])
            docs = loader.load()
            
            # Add metadata for each document
            for doc in docs:
                doc.metadata['source_type'] = 'website'
                doc.metadata['url'] = url
                doc.metadata['loaded_at'] = datetime.now().isoformat()
                
            web_documents.extend(docs)
            logger.info(f"  ✅ Successfully loaded {len(docs)} document(s) from {url}")
            
            # Log details for this URL's documents
            if docs:
                logger.info(f"  📊 Document details from {url}:")
                for j, doc in enumerate(docs, 1):
                    title = doc.metadata.get('title', 'No title')
                    content_length = len(doc.page_content)
                    logger.info(f"    Doc {j}: '{title}' ({content_length:,} chars)")
            
        except Exception as e:
            logger.error(f"  ❌ Error loading {url}: {e}")
            continue
    
    logger.info(f"✅ Successfully loaded {len(web_documents)} total website documents")
    
    # Log detailed website document information
    if web_documents:
        log_document_details(web_documents, 'website')
    
    return web_documents

def log_chunk_details(chunks: List[Document]):
    """Log information about document chunks"""
    logger.info(f"\n{'='*60}")
    logger.info("DOCUMENT CHUNK ANALYSIS")
    logger.info(f"{'='*60}")
    
    # Group chunks by source type
    pdf_chunks = [chunk for chunk in chunks if chunk.metadata.get('source_type') == 'pdf']
    web_chunks = [chunk for chunk in chunks if chunk.metadata.get('source_type') == 'website']
    
    logger.info(f"📊 Chunk breakdown:")
    logger.info(f"  - PDF chunks: {len(pdf_chunks)}")
    logger.info(f"  - Website chunks: {len(web_chunks)}")
    logger.info(f"  - Total chunks: {len(chunks)}")
    
    # Show chunk size statistics
    chunk_sizes = [len(chunk.page_content) for chunk in chunks]
    if chunk_sizes:
        avg_size = sum(chunk_sizes) / len(chunk_sizes)
        min_size = min(chunk_sizes)
        max_size = max(chunk_sizes)
        
        logger.info(f"📏 Chunk size statistics:")
        logger.info(f"  - Average: {avg_size:.0f} characters")
        logger.info(f"  - Minimum: {min_size} characters")
        logger.info(f"  - Maximum: {max_size} characters")
    
    # Show sample chunks from each source type
    logger.info(f"\n📝 Sample chunks:")
    
    if pdf_chunks:
        logger.info(f"  PDF Sample (from {pdf_chunks[0].metadata.get('source', 'Unknown')}):")
        preview = pdf_chunks[0].page_content[:150].replace('\n', ' ').strip()
        logger.info(f"    '{preview}...'")
    
    if web_chunks:
        logger.info(f"  Website Sample (from {web_chunks[0].metadata.get('url', 'Unknown')}):")
        preview = web_chunks[0].page_content[:150].replace('\n', ' ').strip()
        logger.info(f"    '{preview}...'")

def clear_existing_data():
    """Optional: Clear existing data from the vector store"""
    logger.info("🗑️  Attempting to clear existing data from vector store...")
    try:
        response = supabase.table("documents").delete().neq("id", "").execute()
        deleted_count = len(response.data) if response.data else 0
        logger.info(f"✅ Cleared {deleted_count} existing records from vector store")
    except Exception as e:
        logger.warning(f"⚠️  Could not clear existing data: {e}")

def main():
    logger.info("🚀 Starting Enhanced RAG Ingestion Process")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    logger.info(f"⏰ Process started at: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # CONFIGURE YOUR WEBSITE URLs HERE
    website_urls = [
        "https://www.arm.com.ng/",
        "https://www.arm.com.ng/about-us/",
        "https://www.arm.com.ng/investment-managers/about/",
        "https://www.arm.com.ng/investment-managers/mutual-fund/",
        "https://www.arm.com.ng/investment-managers/im-private-wealth/",
        "https://www.arm.com.ng/investment-managers/institutional-asset-management/",
    ]
    
    logger.info(f"🌐 Configured {len(website_urls)} website URLs for loading")
    
    # Optional: Clear existing data (uncomment if you want to start fresh)
    # clear_existing_data()
    
    # Load PDF documents
    pdf_documents = load_pdfs()
    
    # Load website documents
    web_documents = load_websites(website_urls)
    
    # Combine all documents
    all_documents = pdf_documents + web_documents
    
    if not all_documents:
        logger.error("❌ No documents loaded! Please check your PDF directory and website URLs.")
        return
    
    logger.info(f"\n📊 FINAL DOCUMENT SUMMARY:")
    logger.info(f"  - PDF documents: {len(pdf_documents)}")
    logger.info(f"  - Website documents: {len(web_documents)}")
    logger.info(f"  - Total documents: {len(all_documents)}")
    
    # Split documents into chunks
    logger.info("\n🔄 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    
    chunks = text_splitter.split_documents(all_documents)
    logger.info(f"✅ Created {len(chunks)} document chunks")
    
    # Log detailed chunk analysis
    log_chunk_details(chunks)
    
    # Store chunks in vector store
    logger.info("\n💾 Storing chunks in Supabase vector database...")
    try:
        vector_store = SupabaseVectorStore.from_documents(
            chunks,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=1000,
        )
        
        logger.info("✅ Documents successfully stored in vector database!")
        
        # Test retrieval
        logger.info("\n🔍 Testing retrieval...")
        test_results = vector_store.similarity_search("introduction", k=3)
        
        logger.info(f"🎯 Test search returned {len(test_results)} results:")
        for i, doc in enumerate(test_results, 1):
            source_type = doc.metadata.get('source_type', 'unknown')
            source = doc.metadata.get('url' if source_type == 'website' else 'source', 'N/A')
            content_preview = doc.page_content[:100].replace('\n', ' ').strip()
            logger.info(f"  {i}. {source_type.upper()}: {source}")
            logger.info(f"     Preview: {content_preview}...")
        
    except Exception as e:
        logger.error(f"❌ Error storing documents: {e}")
        return
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info(f"\n🎉 Ingestion process completed successfully!")
    logger.info(f"⏱️  Total processing time: {duration}")
    logger.info(f"📁 Log file created with detailed information")
    logger.info("Your RAG system now includes both PDF and website content.")

if __name__ == "__main__":
    main()