from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a YouTube video assistant.

Answer the user's question using the provided video context
and conversation history.

Rules:
- Use the video context as the primary source.
- Do not invent information that is not supported by the video.
- If the answer is not available in the video context, clearly say so.
- Use conversation history to understand follow-up questions.
- Keep answers clear and concise.

Video Context:
{context}
"""
    ),
    (
        "placeholder",
        "{chat_history}"
    ),
    (
        "human",
        "{question}"
    )
])


def create_rag_chain():

    return llm, prompt