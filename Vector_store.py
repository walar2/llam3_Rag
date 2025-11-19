# chunk_store.py
import ollama
import chromadb
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document  # for .docx files

# --------------------------
# CONFIG
# --------------------------
# Folder where Chroma will store its data (in project root)
PERSIST_DIR = "C:\\Users\\minkh\\Desktop\\Persnl_11\\llama3-chatqa+RAG\\Chroma_db"          # <-- your Chroma folder
COLLECTION_NAME = "my_docs"

# folders where your docs live (relative to project root)
def build_vector_store():
    print("[INFO] Starting vector store build")
    print(f"[INFO] Using Chroma directory: {PERSIST_DIR}")

    # 1) connect to Chroma
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # 2) load docs using your existing code
    docs = load_project_docs()
    print(f"[INFO] Loaded {len(docs)} documents")

    if not docs:
        print("[ERROR] No documents loaded. Check DOC_DIRS in textsplitter.py.")
        return

    chunk_counter = 0

    for doc_idx, doc in enumerate(docs):
        # 3) split into chunks with your RecursiveCharacterTextSplitter
        chunks = text_splitter.split_text(doc)
        print(f"[INFO] Doc {doc_idx}: {len(chunks)} chunks")

        for chunk in chunks:
            try:
                # 4) embed each chunk with nomic-embed-text via Ollama
                resp = ollama.embed(
                    model="nomic-embed-text",
                    input=chunk,
                )
                embedding = resp["embeddings"][0]
            except Exception as e:
                print(f"[ERROR] Embedding failed for doc{doc_idx}, chunk{chunk_counter}: {e}")
                return

            chunk_id = f"doc{doc_idx}_chunk{chunk_counter}"

            # 5) store in Chroma
            collection.add(
                ids=[chunk_id],
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source_doc": doc_idx}],
            )
            chunk_counter += 1

    total = collection.count()
    print(f"[INFO] Indexed {chunk_counter} chunks in this run.")
    print(f"[INFO] Collection '{COLLECTION_NAME}' now has {total} vectors.")


if __name__ == "__main__":
    build_vector_store()