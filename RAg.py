# RAG.py
import chromadb
import ollama

PERSIST_DIR = "C:\\Users\\minkh\\Desktop\\Persnl_11\\llama3-chatqa+RAG\\Chroma_db"
COLLECTION_NAME = "my_docs"

# 1) Connect to the same Chroma DB + collection
client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_collection(name=COLLECTION_NAME)  # same name as indexer


def retrieve_context(query: str, k: int = 4):
    # 2) Embed the question
    resp = ollama.embed(
        model="nomic-embed-text",
        input=query,
    )
    query_emb = resp["embeddings"][0]

    # 3) Query the vector store (similarity search)
    results = collection.query(
        query_embeddings=[query_emb],
        n_results=k,
    )

    # results["documents"] is a list of lists: one list per query
    context_chunks = results["documents"][0]  # top-k chunks
    return context_chunks


def build_prompt(context_chunks, question: str) -> str:
    context_text = "\n\n".join(context_chunks)
    return f"""
You are an assistant that answers questions using ONLY the provided context.
If the answer is not in the context, say you don't know.

Context:
{context_text}

Question: {question}

Answer clearly and concisely:
""".strip()


def ask(question: str):
    # 4) Retrieve chunks from Chroma
    context_chunks = retrieve_context(question, k=4)

    # (optional) safety: if nothing found
    if not context_chunks:
        print("No relevant context found in the vector store.")
        return

    # 5) Build prompt
    prompt = build_prompt(context_chunks, question)

    # 6) Call llama3-chatqa with question + retrieved context
    response = ollama.chat(
        model="llama3-chatqa:8b",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful QA assistant using retrieved context.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )

    answer = response["message"]["content"]

    print("Q:", question)
    print("\n--- Answer ---\n")
    print(answer)


if __name__ == "__main__":
    print("RAG chat. Type 'exit' or 'quit' to stop.")

    while True:
        question = input("\nYour question: ")

        if question.strip().lower() in {"exit", "quit"}:
            print("Goodbye 👋")
            break

        ask(question)   # call your ask() function with whatever you typed
