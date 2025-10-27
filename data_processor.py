# data_processor.py

import os
# Document loading from PDF
from langchain_community.document_loaders import PyPDFDirectoryLoader 
# Splitting text
from langchain.text_splitter import RecursiveCharacterTextSplitter
# Open-source embeddings (HuggingFace)
from langchain_community.embeddings import HuggingFaceEmbeddings 
# Vector store (ChromaDB)
from langchain_community.vectorstores import Chroma

# CRITICAL IMPORT: This MUST match the variable names in config.py
from config import DATA_PATH, CHROMA_PATH, EMBEDDING_MODEL_NAME

def load_documents():
    """Load PDF files from the specified data path."""
    print(f"Loading documents from {DATA_PATH}...")
    loader = PyPDFDirectoryLoader(DATA_PATH)
    documents = loader.load()
    print(f"Loaded {len(documents)} document pages.")
    return documents

def split_text(documents):
    """Split documents into smaller, indexable chunks."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200, 
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split documents into {len(chunks)} chunks.")
    return chunks

def create_database(chunks):
    """Create embeddings and store them in the Chroma vector database."""
    print(f"Creating embeddings using {EMBEDDING_MODEL_NAME}...")
    
    # Initialize the open-source embedding model
    embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    
    # Create and persist the Vector Store
    db = Chroma.from_documents(
        chunks,
        embedding_function,
        persist_directory=CHROMA_PATH
    )
    db.persist()
    print(f"Successfully created Chroma database at {CHROMA_PATH}")
    return db

def get_db():
    """Checks if DB exists, if not, creates it. Returns the Chroma DB instance."""
    if not os.path.exists(CHROMA_PATH):
        print("Database not found. Starting indexing process...")
        documents = load_documents()
        chunks = split_text(documents)
        db = create_database(chunks)
        return db
    else:
        # Load the existing database
        print(f"Loading existing Chroma database from {CHROMA_PATH}")
        embedding_function = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        return db

if __name__ == "__main__":
    db = get_db()
    print("Data processing complete.")