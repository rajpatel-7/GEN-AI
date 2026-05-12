from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
)
text = [
    "Hello world",
    "How are you doing today?",
    "I am doing well, thank you!"
]
vectors = embedding.embed_documents(text)
print(vectors)