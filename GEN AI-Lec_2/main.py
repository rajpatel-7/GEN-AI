from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True},
    cache_folder="./embeddings_cache"
)

vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

retriever = vectorstore.as_retriever(
    search_type="mmr", 
    search_kwargs={"k": 5, "fetch_k": 10, "lambda_mult": 0.5}
)
llm = ChatMistralAI(model_name="mistral-small-2506", temperature=0.9, max_tokens=2048)
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that helps answer questions based on the following context:,say:"i couldn't find any relevant information in the context." if you can't find any relevant information in the context."""),
    ("user", """context: {context}\n\nQuestion: {question}"""),
]) 

print("Rag system is ready! Please enter your question:")
print("You can type 'exit' to quit the program.")
while True:
    query = input("You: ")
    if query.lower() == "exit":
        print("Exiting the program. Goodbye!")
        break
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    finalprompt = prompt.format(context=context, question=query)
    response = llm.invoke(finalprompt)
    print(f"Assistant: {response.content}")
