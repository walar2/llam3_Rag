# text_splitter.py

from pathlib import Path
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from docx import Document  # for reading .docx (Word) files

# ==============================
# 1) DIRECTORIES FOR YOUR DOCS
# ==============================
# Add all the folders you want this project to read from.
DOC_DIRS = [
   r"C:\Users\minkh\Desktop\Persnl_11\llama3-chatqa+RAG\New folder"]


def read_txt_file(path: Path) -> str:
    """Read a .txt file as UTF-8 text."""
    return path.read_text(encoding="utf-8")


def read_docx_file(path: Path) -> str:
    """Read a .docx (Word) file as plain text."""
    doc = Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    # join paragraphs with newlines
    return "\n".join(paragraphs)


def load_project_docs() -> List[str]:
    """
    Load all .txt and .docx files from the directories listed in DOC_DIRS.
    Returns a list of text strings, one per file.
    """
    docs: List[str] = []

    for dir_path in DOC_DIRS:
        folder = Path(dir_path)
        if not folder.exists():
            print(f"[WARN] Folder does not exist, skipping: {folder}")
            continue

        # iterate over all files in the folder
        for path in folder.iterdir():
            if not path.is_file():
                continue

            suffix = path.suffix.lower()

            try:
                if suffix == ".txt":
                    text = read_txt_file(path)
                    docs.append(text)
                    print(f"[INFO] Loaded TXT:   {path}")

                elif suffix == ".docx":
                    text = read_docx_file(path)
                    docs.append(text)
                    print(f"[INFO] Loaded DOCX:  {path}")

                else:
                    # ignore other file types
                    continue

            except Exception as e:
                print(f"[ERROR] Failed to read {path}: {e}")

    if not docs:
        print("[WARN] No .txt or .docx files found in any DOC_DIRS.")

    return docs


# ==============================
# 2) TEXT SPLITTER
# ==============================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,   # characters per chunk
    chunk_overlap=200, # characters of overlap
    # separators can be customized if needed
)
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
    # Quick test if you run this file directly
    docs = load_project_docs()
    print(f"\nTotal docs loaded: {len(docs)}")
    build_vector_store()

    if docs:
        chunks = text_splitter.split_text(docs[0])
        print(f"First doc length: {len(docs[0])} characters")
        print(f"Chunks from first doc: {len(chunks)}")
