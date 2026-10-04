from youtube_transcript_api import YouTubeTranscriptApi
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter



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

    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i + 1} ---")
        print(chunk.page_content[:500])
        print("\nMetadata:", chunk.metadata)