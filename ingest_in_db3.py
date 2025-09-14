# import basics
import os
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

# initiate supabase db
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# initiate embeddings model
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

def load_pdfs():
    """Load PDF documents from the documents folder"""
    print("Loading PDF documents...")
    try:
        loader = PyPDFDirectoryLoader("documents")
        documents = loader.load()
        
        # Add source type metadata
        for doc in documents:
            doc.metadata['source_type'] = 'pdf'
            
        print(f"✅ Successfully loaded {len(documents)} PDF documents")
        return documents
    except Exception as e:
        print(f"❌ Error loading PDFs: {e}")
        return []

def load_websites(urls: List[str]):
    """Load content from website URLs"""
    if not urls:
        print("No website URLs provided, skipping...")
        return []
        
    print(f"Loading {len(urls)} website(s)...")
    web_documents = []
    
    for url in urls:
        try:
            print(f"  Loading: {url}")
            loader = WebBaseLoader([url])
            docs = loader.load()
            
            # Add metadata for each document
            for doc in docs:
                doc.metadata['source_type'] = 'website'
                doc.metadata['url'] = url
                
            web_documents.extend(docs)
            print(f"  ✅ Successfully loaded {len(docs)} document(s) from {url}")
            
        except Exception as e:
            print(f"  ❌ Error loading {url}: {e}")
            continue
    
    print(f"✅ Successfully loaded {len(web_documents)} total website documents")
    return web_documents

def clear_existing_data():
    """Optional: Clear existing data from the vector store"""
    try:
        response = supabase.table("documents").delete().neq("id", "").execute()
        print("✅ Cleared existing data from vector store")
    except Exception as e:
        print(f"Warning: Could not clear existing data: {e}")

def main():
    print("🚀 Starting Enhanced RAG Ingestion Process")
    print("=" * 50)
    
    # CONFIGURE YOUR WEBSITE URLs HERE
    website_urls = [
       "https://www.arm.com.ng/",
    "https://www.arm.com.ng/about-us/",
    "https://www.arm.com.ng/investment-managers/about/",
    "https://www.arm.com.ng/investment-managers/mutual-fund/",
    "https://www.arm.com.ng/investment-managers/im-private-wealth/",
    "https://www.arm.com.ng/investment-managers/institutional-asset-management/",
        
        # # Example URLs for testing (replace with your own)
        # "https://python.langchain.com/docs/introduction/",
        # "https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview",
    ]
    
    # Optional: Clear existing data (uncomment if you want to start fresh)
    # clear_existing_data()
    
    # Load PDF documents
    pdf_documents = load_pdfs()
    
    # Load website documents
    web_documents = load_websites(website_urls)
    
    # Combine all documents
    all_documents = pdf_documents + web_documents
    
    if not all_documents:
        print("❌ No documents loaded! Please check your PDF directory and website URLs.")
        return
    
    print(f"\n📊 Document Summary:")
    print(f"  - PDF documents: {len(pdf_documents)}")
    print(f"  - Website documents: {len(web_documents)}")
    print(f"  - Total documents: {len(all_documents)}")
    
    # Split documents into chunks
    print("\n🔄 Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(all_documents)
    
    print(f"✅ Created {len(docs)} document chunks")
    
    # Show breakdown of chunks by source type
    pdf_chunks = sum(1 for doc in docs if doc.metadata.get('source_type') == 'pdf')
    web_chunks = sum(1 for doc in docs if doc.metadata.get('source_type') == 'website')
    
    print(f"  - PDF chunks: {pdf_chunks}")
    print(f"  - Website chunks: {web_chunks}")
    
    # Store chunks in vector store
    print("\n💾 Storing chunks in Supabase vector database...")
    try:
        vector_store = SupabaseVectorStore.from_documents(
            docs,
            embeddings,
            client=supabase,
            table_name="documents",
            query_name="match_documents",
            chunk_size=1000,
        )
        
        print("✅ Documents successfully stored in vector database!")
        
        # Test retrieval
        print("\n🔍 Testing retrieval...")
        test_results = vector_store.similarity_search("introduction", k=3)
        
        print(f"Test search returned {len(test_results)} results:")
        for i, doc in enumerate(test_results, 1):
            source_type = doc.metadata.get('source_type', 'unknown')
            source = doc.metadata.get('url' if source_type == 'website' else 'source', 'N/A')
            print(f"  {i}. {source_type.upper()}: {source}")
        
    except Exception as e:
        print(f"❌ Error storing documents: {e}")
        return
    
    print("\n🎉 Ingestion process completed successfully!")
    print("Your RAG system now includes both PDF and website content.")

if __name__ == "__main__":
    main()