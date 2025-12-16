import os
import time
from typing import Dict

from dotenv import load_dotenv
from google import genai

from app.rag.EmbeddingManager import embedding_manager
from app.rag.Retriever import Retriever
from app.rag.VectorStore import vector_store

# load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

retriever = Retriever(vector_store, embedding_manager)
llm_client = genai.Client()


def simple_rag(query: str, file_id: str) -> Dict[str, str | int | None]:
    """
    Simple RAG implementation using Google GenAI and vector store.

    Args:
        query (str): The user's question.

    Returns:
        str: The answer to the user's question in JSON format.
    """

    # retrieve relevant documents from vector store
    # as additional context for the LLM
    relevant_documents = retriever.retrieve(query, 3, 0.0, file_id)
    context = "\n\n".join([doc["content"] for doc in relevant_documents])

    if not context:
        return {"error": "No relevant context is found to answer the question."}

    # generate the answer with the LLM
    prompt = f"""Use the following context to answer the question concisely.

    context:
    {context}

    question: {query}

    answer:
    """
    generation_start_time = time.time()
    response = llm_client.models.generate_content(
        model="gemini-2.5-flash", contents=prompt
    )
    answer = response.text or ""

    generation_duration = time.time() - generation_start_time
    print(f"Generation duration: {generation_duration:.4f} seconds")

    usage = response.usage_metadata
    input_token = getattr(usage, "prompt_token_count", None)
    output_token = getattr(usage, "candidates_token_count", None)
    total_token = getattr(usage, "total_token_count", None)

    return {
        "answer": answer,
        "input_tokens": input_token,
        "output_tokens": output_token,
        "total_tokens": total_token,
    }
