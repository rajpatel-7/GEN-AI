import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq

load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Funny AI Assistant",
    page_icon="🤖",
    layout="centered"
)

# Custom CSS for attractive styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stChatMessage {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stChatMessage p {
        color: #1a1a1a !important;
        font-size: 16px !important;
        line-height: 1.6 !important;
    }
    [data-testid="stChatMessageContent"] {
        color: #1a1a1a !important;
    }
    .stTextInput > div > div > input {
        border-radius: 20px;
        border: 2px solid #667eea;
        padding: 10px 20px;
        color: #1a1a1a;
    }
    .title-container {
        background: rgba(255, 255, 255, 0.95);
        padding: 20px;
        border-radius: 15px;
        margin-bottom: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Title with custom styling
st.markdown("""
    <div class="title-container">
        <h1 style="color: #667eea; margin: 0;">🤖 Funny AI Assistant</h1>
        <p style="color: #764ba2; margin: 5px 0 0 0;">Powered by Llama 3.3</p>
    </div>
    """, unsafe_allow_html=True)

# Initialize the model
@st.cache_resource
def load_model():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0.7,
    )

model = load_model()

# Initialize session state for history
if "history" not in st.session_state:
    st.session_state.history = [
        SystemMessage(content="You are a funny ai assistant.")
    ]

# Display chat messages (skip SystemMessage)
for message in st.session_state.history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.write(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🤖"):
            st.write(message.content)

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message to history
    st.session_state.history.append(HumanMessage(content=prompt))
    
    # Display user message
    with st.chat_message("user", avatar="👤"):
        st.write(prompt)
    
    # Get bot response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking..."):
            response = model.invoke(st.session_state.history)
            st.write(response.content)
    
    # Add bot response to history
    st.session_state.history.append(AIMessage(content=response.content))

# Sidebar with info
with st.sidebar:
    st.markdown("### 💬 Chat Info")
    st.info(f"Messages: {len([m for m in st.session_state.history if not isinstance(m, SystemMessage)])}")
    
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.history = [
            SystemMessage(content="You are a funny ai assistant.")
        ]
        st.rerun()
    
    st.markdown("---")
    st.markdown("### ⚙️ Settings")
    st.markdown("**Model:** Llama 3.3 70B")
    st.markdown("**Temperature:** 0.7")
    st.markdown("**Role:** Funny AI Assistant")