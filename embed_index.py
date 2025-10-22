# embed_index.py

import pickle
from sentence_transformers import SentenceTransformer
from chromadb import Client
from chromadb.config import Settings

# -------------------------------
# Step 0: Ensure no old DB exists
# -------------------------------
# Make sure ./chroma_db does NOT exist, delete it if it does
# You can do manually: rm -rf chroma_db

# -------------------------------
# Step 1: Initialize Chroma client (new style)
# -------------------------------
client = Client(Settings(
    persist_directory="./chroma_db",  # where DB will be stored
    chroma_db_impl="duckdb+parquet"   # new backend
))

# -------------------------------
# Step 2: Load chunked documents
# -------------------------------
with open("splits.pkl", "rb") as f:
    all_chunks = pickle.load(f)
print(f"Loaded {len(all_chunks)} chunks")

# -------------------------------
# Step 3: Define embedding models
# -------------------------------
models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2", "all-distilroberta-v1"]

# -------------------------------
# Step 4: Embed chunks and store in Chroma
# -------------------------------
for model_name in models:
    print(f"\nEmbedding with model: {model_name}")
    model = SentenceTransformer(model_name)
    
    texts = [doc.page_content for doc in all_chunks]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    collection_name = f"pdf_docs_{model_name.replace('-', '_')}"
    collection = client.get_or_create_collection(name=collection_name)
    
    # Add documents to collection
    collection.add(
        documents=texts,
        metadatas=[doc.metadata if hasattr(doc, "metadata") else {"id": str(i)} for i, doc in enumerate(all_chunks)],
        ids=[str(i) for i in range(len(all_chunks))],
        embeddings=embeddings.tolist()
    )
    
    # Persist after each model
    client.persist()
    print(f"Stored embeddings for {model_name} in collection {collection_name}")

# -------------------------------
# Step 5: Optional test query
# -------------------------------
query = "Explain the main topic of the first document"
for model_name in models:
    print(f"\nQuerying model: {model_name}")
    model = SentenceTransformer(model_name)
    query_embedding = model.encode([query])[0]

    collection_name = f"pdf_docs_{model_name.replace('-', '_')}"
    collection = client.get_collection(name=collection_name)

    results = collection.query(query_embeddings=[query_embedding.tolist()], n_results=3)
    print(f"Top results for {model_name}:")
    for doc in results["documents"][0]:
        print("-", doc[:200], "...")
