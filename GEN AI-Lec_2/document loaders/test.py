from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
splitter=RecursiveCharacterTextSplitter(separators="",chunk_size=10, chunk_overlap=1)
docs = TextLoader("D:\\MLDL\\GEN Ai-2\\document loaders\\text.txt").load()
chunks = splitter.split_documents(docs)
for i in chunks:
    print(i.page_content)