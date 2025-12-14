import time
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Handles document embedding generation using HuggingFace's SentenceTransformer"""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding manager.

        Args:
            model_name (str): The name of the SentenceTransformer model to use for embedding generation.
        """
        self.model_name = model_name
        self.model = None  # will be initialized in the `_load_model()`
        self._load_model()

    def _load_model(self):
        """Load the SentenceTransformer model."""
        try:
            start_time = time.time()
            print(f"Loading the embedding model {self.model_name}...")
            self.model = SentenceTransformer(self.model_name)
            print(
                f"Model loaded successfully. Time taken: {time.time() - start_time:.2f} seconds"
            )
        except Exception as e:
            print(f"Error loading model {self.model_name}: {str(e)}")
            raise

    def generate_embeddings(self, list_text: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts.

        Args:
            list_text (List[str]): List of texts to generate embeddings for.

        Returns:
            np.ndarray: Array of embeddings.
        """
        if not self.model:
            raise ValueError("Model not loaded.")

        start_time = time.time()
        print(f"Generating embeddings for {len(list_text)} text lists...")
        embeddings = self.model.encode(list_text)
        print(f"Generated embeddings with shape: {embeddings.shape}")
        print(f"Time taken: {time.time() - start_time:.2f} seconds")

        return embeddings


embedding_manager = EmbeddingManager()
