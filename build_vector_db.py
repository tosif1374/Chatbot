import os
import shutil
from tqdm import tqdm

from RAG_with_TDS import clean_docs
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS


VECTOR_DB_PATH = "vector_db"
PDF_PATH = "Machine-Learning-Systems.pdf"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
BATCH_SIZE = 200


# delete old db
if os.path.exists(VECTOR_DB_PATH):
    shutil.rmtree(VECTOR_DB_PATH)


# load pdf
pdf_loader = PyPDFLoader(PDF_PATH)
pdf_docs = pdf_loader.load()


# combine pdf + web docs
all_docs = pdf_docs + clean_docs


# split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP
)

docs = splitter.split_documents(all_docs)

print(f"Total chunks: {len(docs)}")


# embeddings
embeddings = OllamaEmbeddings(model="nomic-embed-text")


# build faiss
vectorstore = None

for i in tqdm(range(0, len(docs), BATCH_SIZE), desc="Embedding"):
    
    batch = docs[i:i+BATCH_SIZE]

    if vectorstore is None:
        vectorstore = FAISS.from_documents(batch, embeddings)
    else:
        vectorstore.add_documents(batch)


vectorstore.save_local(VECTOR_DB_PATH)

print("Vector DB created successfully")