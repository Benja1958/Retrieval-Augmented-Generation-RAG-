import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def load_documents(path: str):
    """
    Load MS MARCO documents from a JSON array:

    [
      {"id": 0, "text": "..."},
      {"id": 1, "text": "..."},
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    doc_ids = [item["id"] for item in data]
    texts = [item["text"] for item in data]
    return doc_ids, texts


def encode_documents(docs, batch_size: int = 64):
    """
    Encode all documents using the BGE model (BAAI/bge-base-en-v1.5).

    Returns: numpy array of shape (N, 768), dtype float32.
    """
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"Loading encoder model: {model_name}")
    model = SentenceTransformer(model_name)

    all_embeddings = []

    for i in tqdm(range(0, len(docs), batch_size), desc="Encoding documents"):
        batch = docs[i:i + batch_size]
        emb = model.encode(
            batch,
            normalize_embeddings=True,      # recommended for BGE
            show_progress_bar=False,
        )
        emb = emb.astype("float32")         # FAISS expects float32
        all_embeddings.append(emb)

    embeddings = np.vstack(all_embeddings)
    return embeddings


def save_preprocessed(doc_ids, docs, embeddings, output_path: str):
    """
    Save in the required JSON format:

    [
      {"id": 0, "text": "...", "embedding": [ ... ]},
      ...
    ]
    """
    assert len(doc_ids) == len(docs) == embeddings.shape[0]

    data = []
    for doc_id, text, emb in zip(doc_ids, docs, embeddings):
        item = {
            "id": doc_id,                  # keep original id from dataset
            "text": text,
            "embedding": emb.tolist(),     # numpy -> Python list
        }
        data.append(item)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(data)} documents to {output_path}")


def main():
    input_path = Path("documents.json")
    output_path = Path("preprocessed_documents.json")

    print(f"Loading documents from {input_path} ...")
    doc_ids, docs = load_documents(str(input_path))
    print(f"Loaded {len(docs)} documents.")

    print("Encoding documents into 768-d embeddings ...")
    embeddings = encode_documents(docs)
    print(f"Embeddings shape: {embeddings.shape}")

    print("Saving preprocessed JSON ...")
    save_preprocessed(doc_ids, docs, embeddings, str(output_path))


if __name__ == "__main__":
    main()
