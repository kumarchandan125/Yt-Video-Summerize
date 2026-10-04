import time
from langchain_chroma import Chroma
from embedding_model.embeddings import get_embedding_model

def create_vector_store(chunks):
    embeddings= get_embedding_model()


    vector_store=Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="youtube_video",
        persist_directory="./chroma_db"
    )
    batch_size=10
    for i in range(0,len(chunks),batch_size):
        batch=chunks[i:i+batch_size]

        print(
            f"Processing chunks "
            f"{i + 1}-{i + len(batch)} "
            f"of {len(chunks)}"
        )
        vector_store.add_documents(batch)

        print("Batch stored successfully ✅")

        if i + batch_size < len(chunks):
            print("Waiting before next batch...")
            time.sleep(10)

    return vector_store