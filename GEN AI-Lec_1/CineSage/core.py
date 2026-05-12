from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()
from langchain_mistralai import ChatMistralAI
model = ChatMistralAI(model="mistral-small", temperature=0.7, max_tokens=2048)

prompt = ChatPromptTemplate.from_messages([
    ("system", '''You are an advanced movie information extraction assistant.

Analyze the provided movie paragraph carefully and extract all useful information in a clean, well-structured human-readable format.

Rules:

* Extract only information present or strongly implied.
* Do NOT hallucinate missing facts.
* If some information is unavailable, write "Not Mentioned".
* Keep the response organized and professional.
* Generate a short and engaging quick summary.
* Detect themes, tone, audience, and important cinematic details.
* If multiple movies are mentioned, separate them clearly.

Use the following format exactly:

========================
🎬 MOVIE INFORMATION
====================

Movie Name:
Alternative Titles:
Release Year:
Genres:
Language:
Country:
Runtime:
Director:
Writers:
Producers:

⭐ CAST

* Actor Name as Character Name
* Actor Name as Character Name

🎵 CREW DETAILS
Music Composer:
Cinematographer:
Editor:
Production Companies:
Distribution Companies:

📺 AVAILABILITY
Streaming Platforms:

🏆 RECOGNITION
Awards:
Ratings:

* IMDb:
* Rotten Tomatoes:
* Metacritic:

📖 STORY DETAILS
Plot:
Themes:
Keywords:
Setting:
Time Period:
Tone/Mood:
Target Audience:

🔥 EXTRA DETAILS
Franchise:
Sequel/Prequel:
Famous Dialogues:
Notable Scenes:
Content Warnings:

⚡ QUICK SUMMARY
Write a concise 2-4 sentence engaging summary.

📊 EXTRACTION CONFIDENCE
Confidence Score: __/100

Movie Paragraph:
"""
{movie_paragraph}
"""
'''),
    ("user",  """
Extract movie information from the following paragraph.

Paragraph:
{movie_paragraph}
""")
])
movie_paragraph = input("Enter a movie paragraph: ")
final_prompt = prompt.invoke({"movie_paragraph": movie_paragraph})
response = model.invoke(final_prompt)
print(response.content)