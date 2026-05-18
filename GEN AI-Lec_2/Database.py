from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs1= PyPDFLoader(r"C:\Users\DELL G15\Downloads\d2l-en.pdf").load()
splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
chunks = splitter.split_documents(docs1)

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={
        "normalize_embeddings": True
    }
)
vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
