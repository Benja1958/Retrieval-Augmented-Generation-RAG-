from vector_db import VectorDB
from llm_generation import LLMGenerator


def build_augmented_prompt(query: str, retrieved_docs: list[dict]) -> str:
    """
    Component 4: Prompt augmentation.

    Format:
      Question: ...
      Top documents:
      [Document 1 | id=...]
      ...
    """
    parts = []
    parts.append("You are a helpful assistant. Answer the question using ONLY the information in the documents.\n")
    parts.append(f"Question: {query}\n")
    parts.append("Top documents:\n")

    for i, doc in enumerate(retrieved_docs, start=1):
        parts.append(
            f"[Document {i} | id={doc['doc_id']} | distance={doc['distance']:.4f}]\n"
            f"{doc['text']}\n"
        )

    parts.append("\nNow provide a concise, factual answer to the question above.\nAnswer:")
    return "\n".join(parts)


def main():
    # Initialize DB + LLM
    db = VectorDB("preprocessed_documents.json")
    llm = LLMGenerator("tinyllama-1.1b-chat-v0.3.Q4_K_M.gguf")

    while True:
        query = input("\nEnter a question (or 'exit' to quit).\n> ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        # Components 2 + 3: retrieval 
        retrieved_docs = db.search_by_text(query, k=3)

        # Component 4: augmented prompt
        prompt = build_augmented_prompt(query, retrieved_docs)

        # Component 5: LLM generation
        answer = llm.generate(prompt)

        print("=== Answer ===\n")
        print(answer)
        print("\n================\n")


if __name__ == "__main__":
    main()
