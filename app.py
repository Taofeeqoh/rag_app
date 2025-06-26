import os
import streamlit as st
from rag_api import load_and_index_documents, setup_rag

st.set_page_config(page_title="📚 RAG Assistant", page_icon="🤖")

INDEX_PATH = "faiss_index"

if not os.path.exists(INDEX_PATH):
    os.makedirs("documents", exist_ok=True)
    load_and_index_documents()

rag_chain = setup_rag()

st.title("📚 RAG Assistant (GitHub GPT-4o)")
st.write("Upload your .txt files and ask questions about them.")

# ===== Main Chat Section =====
question = st.text_input("Ask your question:")
if st.button("Ask"):
    if question.strip():
        with st.spinner("Thinking..."):
            answer = rag_chain.run(question)
        st.success("Answer:")
        st.write(answer)
    else:
        st.warning("Please enter a question.")

# ===== Sidebar: Upload and Re-index =====
st.sidebar.title("📁 Upload Documents")
uploaded_files = st.sidebar.file_uploader("Upload .txt files", type=["txt"], accept_multiple_files=True)

if st.sidebar.button("Re-index Documents"):
    if uploaded_files:
        os.makedirs("documents", exist_ok=True)
        for file in uploaded_files:
            file_path = os.path.join("documents", file.name)
            with open(file_path, "wb") as f:
                f.write(file.read())
        load_and_index_documents()
        st.sidebar.success("Documents re-indexed!")
    else:
        st.sidebar.warning("Please upload .txt files first.")
