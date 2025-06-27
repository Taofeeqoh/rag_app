import os
import glob
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_index_documents(directory="documents", index_path="faiss_index"):
    file_paths = glob.glob(f"{directory}/*.txt")
    pdf_paths = glob.glob(f"{directory}/*.pdf")
    documents = []

    for path in file_paths:
        loader = TextLoader(path)
        documents.extend(loader.load())

    for path in pdf_paths:
        loader = PyPDFLoader(path)
        documents.extend(loader.load())

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    db = FAISS.from_documents(split_docs, embeddings)
    db.save_local(index_path)

def setup_rag(index_path="faiss_index"):
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    client = OpenAI(
        base_url="https://models.github.ai/inference",
        api_key=GITHUB_TOKEN,
    )

    class GitHubRAG:
        def __init__(self, retriever, client):
            self.retriever = retriever
            self.client = client

        def run(self, question):
            docs = self.retriever.invoke(question)
            context = "\n".join([doc.page_content for doc in docs]) or "No documents found."

            if not context.strip():
                return "I couldn't find any relevant information in the documents to answer your question."

            prompt = f"""
You are a helpful assistant. You should be able to Explain your Context, summarize or answer any question related to your context. Use only the context below to answer the question. 
If the answer is not found in the context, say "I could not find relevant information in the documents."

Context:
{context}

Question:
{question}
"""

            response = self.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that strictly answers based on provided context."},
                    {"role": "user", "content": prompt}
                ],
                model="openai/gpt-4o",
                temperature=0.7,
                max_tokens=1024
            )
            return response.choices[0].message.content

    return GitHubRAG(retriever, client)
