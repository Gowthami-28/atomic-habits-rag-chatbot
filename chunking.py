from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load PDF
pdf_path = "data/book.pdf"
loader = PyPDFLoader(pdf_path)
documents = loader.load()

print(f"Total pages: {len(documents)}")

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = text_splitter.split_documents(documents)

print(f"Total chunks created: {len(chunks)}")
print("\n----- Preview of first chunk -----\n")
print(chunks[0].page_content)