
from dotenv import load_dotenv
load_dotenv()
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chroma_vector_store.vectorstore import create_vector_store
from rag.rag import create_rag_chain


# ---------------------------------------
# YouTube Video ID
# ---------------------------------------

def get_video_id(url: str) -> str:

    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    raise ValueError("Invalid YouTube URL")


# ---------------------------------------
# Get Transcript
# ---------------------------------------

def get_transcript(url: str) -> str:

    video_id = get_video_id(url)

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    return " ".join(
        snippet.text
        for snippet in transcript
    )


# ---------------------------------------
# Create Chunks
# ---------------------------------------

def create_chunks(
    transcript: str,
    video_id: str
):

    document = Document(
        page_content=transcript,
        metadata={
            "source": "youtube",
            "video_id": video_id
        }
    )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    return splitter.split_documents(
        [document]
    )


# ---------------------------------------
# Main
# ---------------------------------------

if __name__ == "__main__":

    # -----------------------------------
    # 1. YouTube URL
    # -----------------------------------

    url = input("Enter YouTube URL: ").strip()

    # -----------------------------------
    # 2. Video ID
    # -----------------------------------

    video_id = get_video_id(url)

    # -----------------------------------
    # 3. Transcript
    # -----------------------------------

    transcript = get_transcript(url)

    # -----------------------------------
    # 4. Create Chunks
    # -----------------------------------

    chunks = create_chunks(
        transcript,
        video_id
    )

    print(f"\nTotal chunks: {len(chunks)}")

    # -----------------------------------
    # 5. Create Chroma Vector Store
    # -----------------------------------

    vector_store = create_vector_store(
        chunks
    )

    print("\nChunks successfully stored in Chroma!")

    # -----------------------------------
    # 6. MMR Retriever
    # -----------------------------------

    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": min(10, len(chunks)),
            "lambda_mult": 0.5
        }
    )

    # -----------------------------------
    # 7. Create RAG Components
    # -----------------------------------

    llm, query_rewrite_prompt, answer_prompt = create_rag_chain()

    # -----------------------------------
    # 8. Chat History
    # -----------------------------------

    chat_history = []

    print("\nYou can now chat with the video.")
    print("Type 'exit' or 'quit' to stop.")

    # -----------------------------------
    # 9. Chat Loop
    # -----------------------------------

    while True:

        question = input("\nYou: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("\nChat ended.")
            break

        if not question:
            continue

        # --------------------------------
        # 10. Rewrite Query
        # --------------------------------

        rewrite_messages = query_rewrite_prompt.invoke(
            {
                "chat_history": chat_history,
                "question": question
            }
        )

        rewrite_response = llm.invoke(
            rewrite_messages
        )

        # Gemini can return either:
        # str OR list of content blocks

        if isinstance(rewrite_response.content, str):

            rewritten_query = (
                rewrite_response.content.strip()
            )

        else:

            rewritten_query = "".join(
                part.get("text", "")
                for part in rewrite_response.content
                if isinstance(part, dict)
            ).strip()

        # Fallback
        if not rewritten_query:
            rewritten_query = question

        # --------------------------------
        # 11. Retrieve Documents
        # --------------------------------

        documents = retriever.invoke(
            rewritten_query
        )

        # --------------------------------
        # 12. Create Context
        # --------------------------------

        context = "\n\n".join(
            document.page_content
            for document in documents
        )

        # --------------------------------
        # 13. Generate Answer
        # --------------------------------

        answer_messages = answer_prompt.invoke(
            {
                "context": context,
                "chat_history": chat_history,
                "question": question
            }
        )

        response = llm.invoke(
            answer_messages
        )

        # --------------------------------
        # 14. Extract Answer
        # --------------------------------

        if isinstance(response.content, str):

            answer = response.content.strip()

        else:

            answer = "".join(
                part.get("text", "")
                for part in response.content
                if isinstance(part, dict)
            ).strip()

        # --------------------------------
        # 15. Display Answer
        # --------------------------------

        print("\nAI:", answer)

        # --------------------------------
        # 16. Save Conversation
        # --------------------------------

        chat_history.append(
            ("human", question)
        )

        chat_history.append(
            ("ai", answer)
        )
