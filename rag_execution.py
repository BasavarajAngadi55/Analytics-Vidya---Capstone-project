# rag_execution.py
"""
RAG Execution Script (LangChain v0.3+ compatible)
-------------------------------------------------
This script sets up and runs a simple Retrieval-Augmented Generation (RAG) workflow
using LCEL (LangChain Expression Language) — the new, stable API for building chains.
"""

# ✅ Modern imports for LangChain v0.3+
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# ✅ Import your configuration and database setup
from config import RAG_PROMPT_TEMPLATE, LLM_MODEL
from data_processor import get_db


# ------------------------------------------------
# 🔧 Setup the RAG Chain
# ------------------------------------------------
def setup_rag_chain(db):
    """
    Sets up the RAG chain using LCEL (LangChain Expression Language).
    This stable method avoids deprecated 'langchain.chains'.
    """

    # 1. Initialize the LLM (connected to local Ollama)
    llm = Ollama(model=LLM_MODEL)

    # 2. Build retriever (fetches top 3 relevant chunks)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # 3. Define the prompt template (injected with retrieved context)
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # 4. Construct the LCEL pipeline:
    # {context: retriever, input: passthrough} → prompt → llm → output parser
    rag_chain = (
        {
            "context": retriever,
            "input": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ------------------------------------------------
# 💬 Query Function
# ------------------------------------------------
def query_rag(rag_chain, db, query_text: str):
    """
    Invokes the RAG chain and displays the final answer along with retrieved sources.
    """
    print(f"\n--- User Query: {query_text} ---")

    # 1. Retrieve source documents (for transparency)
    retriever = db.as_retriever(search_kwargs={"k": 3})
    retrieved_docs = retriever.invoke(query_text)

    # 2. Invoke the LCEL RAG chain to get final answer
    answer = rag_chain.invoke(query_text)

    print("\n--- AI Answer ---")
    print(answer)
    print("-----------------\n")

    # 3. Display document sources
    print("--- Sources Used (Top 3) ---")
    if retrieved_docs:
        for i, doc in enumerate(retrieved_docs, start=1):
            source = doc.metadata.get("source", "Unknown Source")
            page = doc.metadata.get("page", "Unknown Page")
            print(f"Source {i}: File: {source}, Page: {page}")
    print("----------------------------")

    return answer


# ------------------------------------------------
# 🚀 Main Execution
# ------------------------------------------------
if __name__ == "__main__":
    print("--- Starting Research Bot Setup ---")

    # 1. Load or create vector DB
    db = get_db()

    # 2. Setup LCEL-based RAG Chain
    rag_chain = setup_rag_chain(db)

    # 3. Sample query (you can change this)
    sample_query = "What is the primary benefit of using a RAG system over a standalone LLM?"
    query_rag(rag_chain, db, sample_query)
