from dotenv import load_dotenv
load_dotenv()

# from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

# model = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
# response = model.invoke("Why do parrots talk?")
# print(response.content)


from langchain_groq import ChatGroq

model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
)

response = model.invoke("Hello, who are you?")
print(response.content)