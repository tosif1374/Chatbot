from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DB_PATH = "vector_db"

def load_faiss_tutor(
    llm_model: str = "llama3",
    embed_model: str = "nomic-embed-text",
    k: int = 4,
    temperature: float = 0.4,
):
    """
    Loads an already embedded FAISS vector DB
    and returns a tutor-style callable for RAG QA.
    """

    # 🔹 Load embeddings (NO re-embedding)
    embeddings = OllamaEmbeddings(model=embed_model)

    # 🔹 Load FAISS index
    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # 🔹 Load LLM
    llm = Ollama(
        model=llm_model,
        temperature=temperature
    )

    # 🔹 Tutor callable
    def tutor(query: str) -> str:
        docs = retriever.invoke(query)

        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an AI Tutor on an automated learning platform.

Rules:
- Use document context when relevant
- If the context is incomplete, explain using general knowledge
- Never refuse to answer due to missing context
- Teach clearly and progressively

Document Context:
{context}

Question:
{query}

Answer structure:
1. Direct answer
2. Explanation
3. Example (if helpful)
4. Learning note (if beyond document)
"""
        return llm.invoke(prompt)

    return tutor
