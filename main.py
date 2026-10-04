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
    llm,query_rewrite_prompt,answer_prompt=create_rag_chain()

    #  Conversation History

    chat_history = []
    print("\nYou can now chat with the video.")
    print("Type 'exit' or 'quit' to stop.")

    # Chat Loop
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in {"exit", "quit"}:
            print("\nChat ended.") 
            break
        if not question:
            continue
        #Rewrite User Query
        rewrite_messages=query_rewrite_prompt.invoke(
            {
                "chat_history":chat_history,
                "question":question
            }
        )

        rewritten_query=llm.invoke(
            rewrite_messages
        ).content.strip()

     

        documents=retriever.invoke(rewritten_query)

        #Create Context

        context="\n\n".join(
            document.page_content for document in documents
        )

        #Generate Answer
        answer_messages = answer_prompt.invoke( 
            { 
                "context": context, 
                "chat_history": chat_history, 
                "question": question 
            } 
        )
        response=llm.invoke(answer_messages)
        answer=response.content

        #Display Answer

        print("\nAI: ",answer)

        chat_history.append(
            ("human",question)
        )
        chat_history.append(
            ("ai,",answer)
            
        )
        
