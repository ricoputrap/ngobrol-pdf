import os
from pathlib import Path
from typing import List

import numpy as np
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models import File
from app.rag.EmbeddingManager import embedding_manager
from app.rag.VectorStore import vector_store


def parse_pdf_to_documents(file: File) -> list[Document]:
    """
    Loads a local PDF file and converts it into a list of
    LangChain Document objects with one Document per page.

    Args:
        file_url (str): The URL of the PDF file to load from
        the GET /files endpoint (e.g., "uploads/Google Whitepaper - Prompt Engineering.pdf")

    Returns:
        list[Document]: A list of LangChain Document objects with one Document per page
    """

    # construct the file path from the file URL
    file_url = Path(file.url)
    if not os.path.exists(file_url):
        raise FileNotFoundError(f"File not found: {file_url}")

    print(f"Loading PDF file: {file_url}")

    try:
        # initialize the PyMuPDFLoader
        loader = PyMuPDFLoader(file_url)

        # load the document
        documents: List[Document] = loader.load()

        # augment metadata
        for doc in documents:
            doc.metadata["file_id"] = file.id
            doc.metadata["file_name"] = file.name
            # doc.metadata["user_id"] = file.user_id

        print(f"Successfully loaded {len(documents)} pages.")
        return documents

    except Exception as e:
        print(f"Error processing PDF: {e}")
        return []


def chunk_documents(
    documents: List[Document], chunk_size=1000, chunk_overlap=200
) -> List[Document]:
    """
    Splits the documents into smaller chunks of the specified size & overlap.

    Args:
        documents (List[Document]): The list of documents to chunk.
        chunk_size (int): The size of each chunk.
        chunk_overlap (int): The overlap between chunks.

    Returns:
        List[Document]: A list of chunked documents.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # show example of a chunk
    if chunks:
        print(f"Metadata: {chunks[20].metadata}")
        print(f"Content: {chunks[20].page_content}...")

    return chunks


def generate_embeddings(chunks: List[Document]) -> np.ndarray:
    """
    Generate embeddings for the given chunks of text.

    Args:
        chunks (List[Document]): The list of chunks to generate embeddings for.

    Returns:
        np.ndarray: A numpy array of embeddings.
    """
    list_text = [chunk.page_content for chunk in chunks]
    embeddings = embedding_manager.generate_embeddings(list_text)
    return embeddings


def ingest_file(file: File):
    documents = parse_pdf_to_documents(file)
    chunks = chunk_documents(documents)
    embeddings = generate_embeddings(chunks)
    vector_store.add_documents(chunks, embeddings)

    print("Successfully ingested file.")
