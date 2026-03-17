from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DB_PATH = "vector_db"

def load_faiss_tutor(
    llm_model: str = "llama3",
    embed_model: str = "nomic-embed-text",
    k: int = 4,
    temperature: float = 0.2,
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
You are an AI Learning Tutor specializing in Machine Learning, Deep Learning, NLP, and LLMs.

STRICT RULES:
1. If the user sends a greeting (hi, hello, hey, etc.), respond ONLY with a friendly welcome message. Do NOT explain any ML concept.
2. If the question is unrelated to ML/DL/NLP/LLMs, politely say you only cover those topics.
3. Never generate an answer if the question is vague or a greeting.
4. Only answer questions clearly related to AI/ML topics.
5. Do not hallucinate — if unsure, say "I don't know" rather than guessing.;

Document Context:
{context}

Question:
{query}

Answer structure:
1. In short
2. Explanation
3. Example (if helpful)
4. Learning note (if beyond document)
"""
        return llm.invoke(prompt)

    return tutor
