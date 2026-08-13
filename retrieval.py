from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Load the existing vector store
embeddings = GoogleGenerativeAIEmbeddings(
    model="gemini-embedding-001",
    google_api_key=api_key
)

vectorstore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embeddings
)

# Create retriever
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Test question
question = "What is Atomic Habits?"

print("Question:", question)
print("\nRetrieving relevant chunks...\n")

relevant_docs = retriever.invoke(question)

for i, doc in enumerate(relevant_docs):
    print(f"----- Chunk {i+1} -----")
    print(doc.page_content[:300])  # print first 300 characters
    print()