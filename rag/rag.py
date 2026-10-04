from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate


llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0
)


# ---------------------------------------
# Query Rewriting Prompt
# ---------------------------------------

query_rewrite_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You rewrite follow-up questions into standalone search queries.

Use the conversation history to understand references such as:
- he
- she
- it
- this
- that
- why
- how
- what about it

Rules:
- Return ONLY the rewritten search query.
- Do not answer the question.
- If the question is already standalone, return it unchanged.
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


# ---------------------------------------
# Answer Prompt
# ---------------------------------------

answer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a YouTube video assistant.

Answer the user's question using the provided video context
and conversation history.

Rules:
- Use the video context as the primary source.
- Do not invent information.
- If the answer is not available in the video context,
  clearly say that it is not available in the video.
- Use conversation history only to understand the user's intent.
- Give a clear and concise answer.

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

    return llm, query_rewrite_prompt, answer_prompt

