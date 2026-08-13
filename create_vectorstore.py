from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os
import time

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load + Split
pdf_path = "data/book.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)
print(f"Total chunks available: {len(chunks)}")

# Embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

print("Creating vector store with 60 chunks...")

# Using fewer chunks to avoid rate limit
vectorstore = Chroma.from_documents(
    documents=chunks[:60],
    embedding=embeddings,
    persist_directory="chroma_db"
)

print("Vector store created successfully with 60 chunks!")