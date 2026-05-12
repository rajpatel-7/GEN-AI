from dotenv import load_dotenv
from langchain_core.messages import  HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()


model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)
print("ChatGroq model loaded. You can start chatting with the bot. Type 'exit' to quit.")
prompt=""
history=[
    SystemMessage(content="You are a funny ai assistant.")   
]
while True:
    prompt=input("YOU:  ")
    if prompt=="exit":
        break
    history.append(HumanMessage(content=prompt))
    response = model.invoke(history)
    history.append(AIMessage(content=response.content))
    print("BOT:", response.content)
print(history)    