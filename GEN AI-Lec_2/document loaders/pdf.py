from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter
docs = PyPDFLoader(r"C:\Users\DELL G15\Downloads\Raj_Velkariya_All.pdf").load()
splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=10)
chunks = splitter.split_documents(docs)
print(chunks[0])