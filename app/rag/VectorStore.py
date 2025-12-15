import os
import uuid
from typing import List

import chromadb
import numpy as np
from langchain_core.documents import Document


class VectorStore:
    """Manages document embeddings in a ChromaDB vector store."""

    def __init__(self, collection_name: str, persist_directory: str):
        """
        Initialize a new VectorStore instance.

        Args:
            collection_name (str): The name of the collection to store embeddings.
            persist_directory (str): The directory to persist the collection.
        """
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.client = None
        self.collection = None
        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client & collection."""
        try:
            # create directory if not exists
            os.makedirs(self.persist_directory, exist_ok=True)

            # init a persistent chromadb client
            self.client = chromadb.PersistentClient(path=self.persist_directory)

            # get or create collection
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"description": "PDF document embeddings for RAG"},
            )

            print(f"Vector store initialized. Collection: {self.collection_name}")
            print(f"Existing documents in collection: {self.collection.count()}")

        except Exception as e:
            print(f"Error initializing vector store: {e}")
            raise

    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """
        Add documents and embeddings to the vector store.

        Args:
            documents (List[Document]): A list of documents to add.
            embeddings (np.ndarray): A 2D array of embeddings corresponding to the documents.
        """
        if len(documents) != len(embeddings):
            raise ValueError("Number of documents and embeddings must match.")

        print(f"Adding {len(documents)} documents to collection...")

        # prepare data for insertion
        list_id = []
        list_metadata = []
        list_content = []
        list_embedding = []

        for i, (doc, embedding) in enumerate(zip(documents, embeddings)):
            # generate unique ID for each document
            doc_id = f"doc_{uuid.uuid4().hex[:8]}_{i}"
            list_id.append(doc_id)

            # add useful metadata
            metadata = dict(doc.metadata)
            metadata["doc_index"] = i
            metadata["content_length"] = len(doc.page_content)
            list_metadata.append(metadata)

            # add content of each document
            list_content.append(doc.page_content)

            # embedding
            list_embedding.append(embedding.tolist())

        # populate data to the vector store
        try:
            if self.collection:
                self.collection.add(
                    ids=list_id,
                    embeddings=list_embedding,
                    metadatas=list_metadata,
                    documents=list_content,
                )

                print(f"Successfully added {len(documents)} documents to vector store.")
                print(f"Total documents in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Error adding documents to vector store: {e}")
            raise


vector_store = VectorStore("pdf_documents", "./data/vector_store")
