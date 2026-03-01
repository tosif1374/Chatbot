import os
from tqdm import tqdm

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS


# ===============================
# CONFIG
# ===============================
PDF_PATH = "Machine-Learning-Systems.pdf"     #CHANGE THIS
VECTOR_DB_PATH = "vector_db"
BATCH_SIZE = 200                   # best for GPU
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# ===============================
# LOAD PDF
# ===============================
print(" Loading PDF...")
loader = PyPDFLoader(PDF_PATH)
documents = loader.load()

print(f" Pages loaded: {len(documents)}")

# ===============================
# SPLIT DOCUMENTS
# ===============================
print(" Splitting into chunks...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

docs = text_splitter.split_documents(documents)
print(f"Total chunks: {len(docs)}")

# ===============================
# EMBEDDINGS (GPU via Ollama)
# ===============================
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# ===============================
# BUILD VECTOR DB (BATCHED)
# ===============================
print(" Building vector database (ONE TIME)...")

vectorstore = None

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding"):
    batch_docs = docs[i:i + BATCH_SIZE]

    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch_docs, embeddings)
    else:
        vectorstore.add_documents(batch_docs)

# ===============================
# SAVE VECTOR DB
# ===============================
vectorstore.save_local(VECTOR_DB_PATH)

print(" DONE: Vector database saved successfully!")
print(f" Location: {VECTOR_DB_PATH}")
