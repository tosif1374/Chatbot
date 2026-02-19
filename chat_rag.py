from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama

DB_PATH = "vector_db"

def load_rag_chain():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    vectorstore = FAISS.load_local(
        DB_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    llm = Ollama(
        model="llama3",
        temperature=0.4
    )

    # ✅ THIS is your tutor brain
    def tutor(query: str):
        docs = retriever.get_relevant_documents(query)
        context = "\n\n".join(doc.page_content for doc in docs)

        prompt = f"""
You are an AI Tutor on an automated learning platform.

Rules:
- Use document context when relevant
- If context is incomplete, explain using general knowledge
- Never refuse to answer
- Teach clearly and progressively

Document Context:
{context}

Question:
{query}

Answer with:
1. Direct answer
2. Explanation
3. Example (if useful)
4. Learning note (if beyond the document)
"""
        return llm.invoke(prompt)

    return tutor
