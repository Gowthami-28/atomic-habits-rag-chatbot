import os
from langchain_community.document_loaders import PyPDFLoader

pdf_path = "data/book.pdf"   # change this to your actual PDF name

print("Current working directory:", os.getcwd())
print("Looking for file at:", pdf_path)
print("File exists:", os.path.exists(pdf_path))

if not os.path.exists(pdf_path):
    print("\nERROR: PDF file not found. Please check the file name and folder.")
else:
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"\nTotal pages loaded: {len(documents)}")
    print("\n----- First page content preview -----\n")
    print(documents[0].page_content[:500])