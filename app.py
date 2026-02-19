import streamlit as st
import time
from faiss_loader import load_faiss_tutor

st.set_page_config(page_title="📘 RAG Chatbot", layout="wide")
st.title("📘 AI Learning Tutor")

@st.cache_resource
def get_tutor():
    return load_faiss_tutor()

tutor = get_tutor()

query = st.text_input("Ask a question:")

if query:
    with st.spinner("Thinking..."):
        start_time = time.time()          # ⏱ start timer
        answer = tutor(query)
        end_time = time.time()            # ⏱ end timer

        latency = end_time - start_time   # seconds

        st.write(answer)
        st.markdown(
            f"⏱ **Response Time:** `{latency:.2f} seconds`"
        )
