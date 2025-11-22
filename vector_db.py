import json
from pathlib import Path

import numpy as np
import faiss
from encode import encode_query


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
        - self.doc_ids     : list[int]
        - self.texts       : list[str]
        - self.embeddings  : np.ndarray of shape (N, 768), float32
        - self.id_to_text  : dict[int, str]
        """
        print(f"Loading preprocessed embeddings from {self.preprocessed_path} ...")
        with open(self.preprocessed_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.doc_ids = []
        self.texts = []
        self.id_to_text = {}
        vectors = []

        for item in data:
            doc_id = item["id"]
            text = item["text"]
            emb = item["embedding"]

            self.doc_ids.append(doc_id)
            self.texts.append(text)
            self.id_to_text[doc_id] = text   # <-- for retrieval
            vectors.append(emb)

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
        # alias for part 2 naming
    
    def search(self, query_embedding, top_k=3):
        return self.search_by_vector(query_embedding, k=top_k)

    
    def search_by_text(self, query: str, k: int = 3):
        """
        High-level search: take a text query, encode it, run FAISS,
        and return top-k (doc_id, text, distance).
        """
        # 1. Encode query into 768-dim embedding
        query_vec = encode_query(query)        

        # 2. Use existing vector search
        distances, doc_ids = self.search_by_vector(query_vec, k=k)

        # 3. Attach document text for each result
        results = []
        for dist, doc_id in zip(distances, doc_ids):
            results.append(
                {
                    "doc_id": int(doc_id),
                    "text": self.id_to_text[doc_id],
                    "distance": float(dist),
                }
            )
        return results
    
    def get_texts(self, doc_ids):
        """
        Given a list of document ids (original IDs), return their texts.
        """
        return [self.id_to_text[doc_id] for doc_id in doc_ids]

    def search_and_retrieve(self, query_embedding: np.ndarray, top_k: int = 3):
        """
        Returns a list of dicts:
        [
          {"doc_id": ..., "text": "...", "distance": ...},
          ...
        ]
        """
        distances, doc_ids = self.search(query_embedding, top_k=top_k)
        texts = self.get_texts(doc_ids)

        results = []
        for doc_id, text, dist in zip(doc_ids, texts, distances):
            results.append(
                {
                    "doc_id": int(doc_id),
                    "text": text,
                    "distance": float(dist),
                }
            )
        return results