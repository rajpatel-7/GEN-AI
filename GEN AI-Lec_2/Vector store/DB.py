from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs1= PyPDFLoader(r"C:\Users\DELL G15\Downloads\API - Project Documentation.pdf").load()
splitter=RecursiveCharacterTextSplitter(separators="",chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs1)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},  
    encode_kwargs={
        "normalize_embeddings": True
    }
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

results = vectorstore.similarity_search("What is project goal?", k=1)
print(results[0].page_content)