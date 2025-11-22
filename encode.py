import numpy as np
from sentence_transformers import SentenceTransformer

# Global cached model so we only load BGE once
_encoder_model = None


def get_encoder():
    """
    Lazily load and cache the BGE encoder model.
    We reuse the same model as in Part 1: BAAI/bge-base-en-v1.5.
    """
    global _encoder_model
    if _encoder_model is None:
        model_name = "BAAI/bge-base-en-v1.5"
        print(f"Loading query encoder model: {model_name}")
        _encoder_model = SentenceTransformer(model_name)
    return _encoder_model


def encode_query(query: str) -> np.ndarray:
    """
    Encode a user query into a 768-dimensional embedding.

    Args:
        query: Natural language question from the user.

    Returns:
        A numpy array of shape (768,), dtype float32.
    """
    model = get_encoder()

    # model.encode returns shape (1, 768) for a single string in a list
    emb = model.encode(
        [query],
        normalize_embeddings=True,   # keep consistent with document embeddings
        show_progress_bar=False,
    )

    # emb shape: (1, 768) -> take first row -> (768,)
    emb = emb.astype("float32")[0]
    return emb


if __name__ == "__main__":
    # Simple CLI test: python encode.py "your question here"
    import sys

    if len(sys.argv) > 1:
        query_text = " ".join(sys.argv[1:])
    else:
        query_text = input("Enter your query: ")

    vec = encode_query(query_text)
    print("Query:", query_text)
    print("Embedding shape:", vec.shape)
    print("dtype:", vec.dtype)
    print("First 10 dimensions:", vec[:10])
