import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from pydantic import BaseModel
from typing import List,Optional
from langchain_core.output_parsers import PydanticOutputParser

class Movie(BaseModel):
    title: str
    release_year: Optional[int]
    genres: List[str]
    director: Optional[str]
    cast: List[str]
    plot_summary: Optional[str]
    ratings: Optional[float]
    
parser = PydanticOutputParser(pydantic_object=Movie)

# Load environment variables
load_dotenv()

# # Page configuration
# st.set_page_config(
#     page_title="Movie Information Extractor",
#     page_icon="🎬",
#     layout="wide"
# )

# # Custom styling
# st.markdown("""
#     <style>
#         .main {
#             padding-top: 2rem;
#         }
#         .stTextArea textarea {
#             font-size: 16px;
#         }
#         .title {
#             text-align: center;
#             font-size: 42px;
#             font-weight: bold;
#             color: #ff4b4b;
#         }
#         .subtitle {
#             text-align: center;
#             font-size: 18px;
#             color: gray;
#             margin-bottom: 30px;
#         }
#     </style>
# """, unsafe_allow_html=True)

# Header
# st.markdown('<div class="title">🎬 Movie Information Extractor</div>', unsafe_allow_html=True)
# st.markdown(
#     '<div class="subtitle">Paste any movie paragraph and extract structured movie details instantly.</div>',
#     unsafe_allow_html=True
# )

# Initialize model
model = ChatMistralAI(
    model="mistral-small",
    temperature=0.7,
    max_tokens=2048
)

prompt= ChatPromptTemplate.from_messages([
    ("system", '''You are an advanced movie information extraction assistant.so extract all useful information in a clean,{format_instructions}'''),
    ("user",  '''{movie_paragraph}''')])
# Prompt template
# prompt = ChatPromptTemplate.from_messages([
#     ("system", '''You are an advanced movie information extraction assistant.

# Analyze the provided movie paragraph carefully and extract all useful information in a clean, well-structured human-readable format.

# Rules:

# * Extract only information present or strongly implied.
# * Do NOT hallucinate missing facts.
# * If some information is unavailable, write "Not Mentioned".
# * Keep the response organized and professional.
# * Generate a short and engaging quick summary.
# * Detect themes, tone, audience, and important cinematic details.
# * If multiple movies are mentioned, separate them clearly.

# Use the following format exactly:

# ========================
# 🎬 MOVIE INFORMATION
# ====================

# Movie Name:
# Alternative Titles:
# Release Year:
# Genres:
# Language:
# Country:
# Runtime:
# Director:
# Writers:
# Producers:

# ⭐ CAST

# * Actor Name as Character Name
# * Actor Name as Character Name

# 🎵 CREW DETAILS
# Music Composer:
# Cinematographer:
# Editor:
# Production Companies:
# Distribution Companies:

# 📺 AVAILABILITY
# Streaming Platforms:

# 🏆 RECOGNITION
# Awards:
# Ratings:

# * IMDb:
# * Rotten Tomatoes:
# * Metacritic:

# 📖 STORY DETAILS
# Plot:
# Themes:
# Keywords:
# Setting:
# Time Period:
# Tone/Mood:
# Target Audience:

# 🔥 EXTRA DETAILS
# Franchise:
# Sequel/Prequel:
# Famous Dialogues:
# Notable Scenes:
# Content Warnings:

# ⚡ QUICK SUMMARY
# Write a concise 2-4 sentence engaging summary.

# 📊 EXTRACTION CONFIDENCE
# Confidence Score: __/100

# Movie Paragraph:
# """
# {movie_paragraph}
# """
# '''),
#     ("user", """
# Extract movie information from the following paragraph.

# Paragraph:
# {movie_paragraph}
# """)
# ])

# User input
# movie_paragraph = st.text_area(
#     "Enter Movie Paragraph:",
#     height=300,
#     placeholder="Paste your movie description or paragraph here..."
# )

# Extract button
# if st.button("Extract Movie Information", use_container_width=True):
#     if movie_paragraph.strip():
#         with st.spinner("Analyzing movie details..."):
movie_paragraph = input("Enter a movie paragraph: ")
final_prompt = prompt.invoke({"movie_paragraph": movie_paragraph,"format_instructions":parser.get_format_instructions()})
response = model.invoke(final_prompt)
print(response.content)

    #     st.markdown("## 📄 Extracted Information")
    #     st.markdown(response.content)
    # else:
    #     st.warning("Please enter a movie paragraph first.") 