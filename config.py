# config.py

# --- File Paths ---
DATA_PATH = "data"
CHROMA_PATH = "chroma_db"

# --- Open-Source Model Names ---
# Open-source embedding model from Hugging Face (no API key needed)
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" 
# Local LLM model (requires Ollama to be installed and running, e.g., 'ollama pull llama2')
LLM_MODEL = "llama2" 

# --- Prompt Template ---
RAG_PROMPT_TEMPLATE = """
You are an expert research paper answer bot. Answer the user's question based ONLY 
on the context provided below. Be concise and accurate.
If the answer is not in the context, clearly state: "The provided documents do not contain the answer."

CONTEXT:
{context}

QUESTION:
{input}
"""