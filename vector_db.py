import json
from pathlib import Path

import numpy as np
import faiss


class VectorDB:
    def __init__(self, preprocessed_path: str = "preprocessed_documents.json"):
        """
        Vector database using FAISS IndexFlatL2.

        - Loads embeddings from preprocessed_documents.json
        - Builds a FAISS index
        """
        self.preprocessed_path = Path(preprocessed_path)
        self._load_data()
        self._build_index()

    def _load_data(self):
        """
        Load preprocessed_documents.json and store:
        - self.doc_ids  : list[int]
        - self.texts    : list[str]
        - self.embeddings : np.ndarray of shape (N, 768), float32
        """
        print(f"Loading preprocessed embeddings from {self.preprocessed_path} ...")
        with open(self.preprocessed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.doc_ids = []
        self.texts = []
        vectors = []

        for item in data:
            self.doc_ids.append(item["id"])
            self.texts.append(item["text"])
            vectors.append(item["embedding"])

        self.embeddings = np.array(vectors, dtype="float32")
        print(f"Loaded {len(self.doc_ids)} embeddings of dim {self.embeddings.shape[1]}.")

    def _build_index(self):
        """
        Build a FAISS IndexFlatL2 over the embeddings.
        """
        d = self.embeddings.shape[1]  
        print(f"Building FAISS IndexFlatL2 with dimension {d} ...")
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.embeddings)
        print(f"Index contains {self.index.ntotal} vectors.")

    def search_by_vector(self, query_vec: np.ndarray, k: int = 5):
        """
        Core search function for Part 1.

        Input:
          - query_vec: numpy array of shape (768,) or (1, 768), float32
          - k: number of nearest neighbors

        Returns:
          - distances: 1D numpy array of length k
          - doc_ids:   list of k document ids (original MS MARCO ids)
        """
        q = np.array(query_vec, dtype="float32")
        if q.ndim == 1:
            q = q.reshape(1, -1)

        
        D, I = self.index.search(q, k)  

        # Map FAISS indices back to original doc ids
        doc_ids = [self.doc_ids[idx] for idx in I[0]]
        return D[0], doc_ids


def self_test(preprocessed_path: str = "preprocessed_documents.json"):
    """
    Part 1 Step 3: sanity check.

    Use the embedding of the first document as the query.
    The top result should be the same document (distance ≈ 0).
    """
    db = VectorDB(preprocessed_path)

    # Query = embedding of document at index 0
    query_vec = db.embeddings[0]
    distances, doc_ids = db.search_by_vector(query_vec, k=5)

    print("\nSelf-test search results for doc index 0:")
    print("Distances:", distances)
    print("Doc IDs:  ", doc_ids)
    print("Expected first doc id:", db.doc_ids[0])

    if doc_ids[0] == db.doc_ids[0] and distances[0] < 1e-5:
        print("✅ Passed: doc 0 is closest to itself with distance ≈ 0.")
    else:
        print("⚠️ Warning: expected the first result to be the same doc with distance ~0.")


if __name__ == "__main__":
    self_test()
