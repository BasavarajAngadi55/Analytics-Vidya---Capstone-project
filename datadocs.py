from langchain_community.document_loaders import PyPDFLoader
import os

# List to hold all document pages
docs = []

data_folder = "data"

for file in os.listdir(data_folder):
    if file.endswith(".pdf"):
        file_path = os.path.join(data_folder, file)
        loader = PyPDFLoader(file_path)
        docs.extend(loader.load())
        print(f"Loaded {file}: {len(loader.load())} pages")

print(f"\nTotal pages loaded: {len(docs)}")

