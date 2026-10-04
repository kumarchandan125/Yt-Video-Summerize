from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from chroma_vector_store.vectorstore import create_vector_store
from rag.rag import create_rag_chain


def get_video_id(url: str) -> str:
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]

    if "watch?v=" in url:
        return url.split("watch?v=")[1].split("&")[0]

    raise ValueError("Invalid YouTube URL")


def get_transcript(url: str) -> str:
    video_id = get_video_id(url)

    api = YouTubeTranscriptApi()

    transcript = api.fetch(video_id)

    text = " ".join(snippet.text for snippet in transcript)

    return text


def creat_chunks(transcript:str,video_id:str):
    document=Document(
        page_content=transcript,
        metadata={
            "source":"youtube",
            "video_id":video_id
        }
    )
    splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
    )
    chunks=splitter.split_documents([document])
    return chunks




if __name__ == "__main__":
    url = input("Enter YouTube URL: ")
    video_id=get_video_id(url)

    transcript = get_transcript(url)
    chunks=creat_chunks(transcript,video_id)

    print(f"\nTotal chunks: {len(chunks)}")
    vector_store= create_vector_store(chunks)

    print("\nChunks successfully stored in Chroma!")

    retriever=vector_store.as_retriever(
        search_kwargs={
            "k":3
        }
    )
    llm,prompt=create_rag_chain()

    chat_history = []
    print("\nYou can now chat with the video.")
    print("Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("\nChat ended.") 
            break
        if not question:
            continue
        documents=retriever.invoke(question)
        context="\n\n".join(
            document.page_content for document in documents
        )
        messages = prompt.invoke( 
            { 
                "context": context, 
                "chat_history": chat_history, 
                "question": question 
            } 
        )
        
