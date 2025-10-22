from langchain.text_splitter import RecursiveCharacterTextSplitter
import pickle

# Import the docs list from datadocs.py
from datadocs import docs

# Create a text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,     # size of each chunk
    chunk_overlap=200    # overlap between chunks
)

# Split all documents
splits = splitter.split_documents(docs)

print(f"Original documents: {len(docs)} pages")
print(f"After splitting: {len(splits)} chunks")

# Save chunks to disk so we can reuse later
with open("splits.pkl", "wb") as f:
    pickle.dump(splits, f)

print("Chunks saved to splits.pkl")
