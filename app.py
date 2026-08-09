import streamlit as st
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Page Config
st.set_page_config(
    page_title="Atomic Habits RAG Chatbot",
    page_icon="📘",
    layout="centered"
)

# Header
st.title("📘 Atomic Habits RAG Chatbot")
st.markdown("Ask questions about the book **Atomic Habits** by James Clear and get answers from the document.")
st.divider()

# Load RAG chain
@st.cache_resource
def load_rag_chain():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="gemini-embedding-001",
        google_api_key=api_key
    )

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Answer the question using only the following context from the book Atomic Habits.
    If the answer is not in the context, say "I don't know based on the provided document."

    Context:
    {context}

    Question: {question}
    """)

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

rag_chain = load_rag_chain()

# Input Section
st.subheader("Ask a Question")
question = st.text_input(
    "Type your question here:",
    placeholder="e.g. What is the 1% improvement rule?"
)

col1, col2 = st.columns([1, 5])
with col1:
    ask_button = st.button("Get Answer", type="primary")

# Answer Section
if ask_button and question:
    with st.spinner("Searching the book and generating answer..."):
        answer = rag_chain.invoke(question)
        st.success("Answer:")
        st.write(answer)

elif ask_button and not question:
    st.warning("Please enter a question first.")

# Footer
st.divider()
st.caption("Built with LangChain • ChromaDB • Gemini • Streamlit")