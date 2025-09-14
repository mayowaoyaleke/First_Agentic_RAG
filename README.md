# Agentic RAG (Retrieval Augmented Generation) with LangChain and Supabase

## Prerequisites
- Python 3.11+

## Setup Instructions

### 1. Create a virtual environment
```bash
python -m venv venv
2. Activate the virtual environment
bash
Copy code
# On Windows
venv\Scripts\activate

# On Mac/Linux
source venv/bin/activate
3. Install libraries
bash
Copy code
pip install -r requirements.txt
4. Create accounts
Create a free account on Supabase

Create an API key on OpenAI

5. Execute SQL queries in Supabase
Run the following SQL query in Supabase:

sql
Copy code
-- Enable the pgvector extension to work with embedding vectors
create extension if not exists vector;

-- Create a table to store your documents
create table
  documents (
    id uuid primary key,
    content text, -- corresponds to Document.pageContent
    metadata jsonb, -- corresponds to Document.metadata
    embedding vector (1536) -- 1536 works for OpenAI embeddings, change if needed
  );

-- Create a function to search for documents
create function match_documents (
  query_embedding vector (1536),
  filter jsonb default '{}'
) returns table (
  id uuid,
  content text,
  metadata jsonb,
  similarity float
) language plpgsql as $$
#variable_conflict use_column
begin
  return query
  select
    id,
    content,
    metadata,
    1 - (documents.embedding <=> query_embedding) as similarity
  from documents
  where metadata @> filter
  order by documents.embedding <=> query_embedding;
end;
$$;
6. Add API keys to .env file
Rename .env.example to .env

Add the API keys for Supabase and OpenAI to the .env file

Executing the Scripts
Open a terminal (e.g., in VS Code) and run:

bash
Copy code
python ingest_in_db.py
python agentic_rag.py
streamlit run agentic_rag_streamlit.py
Sources
Resources used while building this project:

LangChain + Supabase

LangChain + OpenAI Embeddings

OpenAI Embeddings Guide

Kaggle: Document Splitting with LangChain

OpenAI: New Embedding Models & API Updates

Zilliz: Text Embedding 3 Small

less
Copy code
