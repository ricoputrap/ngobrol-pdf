import time
from typing import Any, Dict, List

from app.rag.EmbeddingManager import EmbeddingManager
from app.rag.VectorStore import VectorStore


class Retriever:
    """Handles query-based retrieval of documents from the vector store"""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        """
        Initialize the retriever with a vector store and an embedding manager.

        Args:
            vector_store (VectorStore): The vector store to retrieve documents from.
            embedding_manager (EmbeddingManager): The embedding manager to use for embedding queries.
        """
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
        file_id: str = "",
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from the vector store based on the query.

        Args:
            query (str): The query to retrieve documents for (e.g. questions/commands).
            top_k (int): The top k documents to retrieve that are relevant to the query.
            score_threshold (float): The minimum similarity score required for a document to be considered relevant.
            file_id (str): The ID of the file to retrieve documents from.

        Returns:
            List[Dict[str, Any]]: The retrieved documents as well as some additional information.
        """
        print(f"Retrieving documents for query: {query}")

        # generate embeddings for the query
        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        # preprocess the retrieved documents
        retrieved_docs = []

        try:
            if self.vector_store.collection:
                retrieval_start_time = time.time()
                # retrieve documents from the vector store
                query_results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=top_k,
                    where={"file_id": file_id},
                )
                duration = time.time() - retrieval_start_time
                print(f"Query duration: {duration:.4f} seconds")

                post_retrieval_start_time = time.time()
                # deconstruct query results
                ids = query_results["ids"][0]
                documents, metadatas, distances = [], [], []
                if query_results["documents"] and query_results["documents"][0]:
                    documents = query_results["documents"][0]
                if query_results["metadatas"] and query_results["metadatas"][0]:
                    metadatas = query_results["metadatas"][0]
                if query_results["distances"] and query_results["distances"][0]:
                    distances = query_results["distances"][0]

                for i, (doc_id, doc, metadata, distance) in enumerate(
                    zip(ids, documents, metadatas, distances)
                ):
                    # convert distance to similarity score (ChromaDB uses cosine similarity)
                    similarity_score = 1 - distance

                    if similarity_score > score_threshold:
                        retrieved_docs.append(
                            {
                                "id": doc_id,
                                "content": doc,
                                "metadata": metadata,
                                "similarity_score": similarity_score,
                                "distance": distance,
                                "rank": i + 1,
                            }
                        )

                print(f"Retrieved {len(retrieved_docs)} documents after filtering.")
                post_retrieval_duration = time.time() - post_retrieval_start_time
                print(
                    f"Post-retrieval processing duration: {post_retrieval_duration:.4f} seconds"
                )

        except Exception as e:
            print(f"Error retrieving documents: {e}")

        return retrieved_docs
