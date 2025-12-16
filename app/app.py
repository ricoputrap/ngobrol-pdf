import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile
from sqlmodel import select

from app.models import File, SessionDependency
from app.rag.EmbeddingManager import embedding_manager
from app.rag.ingestion import ingest_file
from app.rag.main import simple_rag
from app.rag.Retriever import Retriever
from app.rag.VectorStore import vector_store
from app.schemas import AskRequest

app = FastAPI()


@app.get("/")
def read_root():
    return {"message": "Hello from ngobrol-pdf!"}


@app.get("/files")
def get_files(session: SessionDependency):
    statement = select(File)
    files = session.exec(statement).all()
    return files


@app.post("/files")
async def upload_file(file: UploadFile, session: SessionDependency):
    # create uploads directory if it doesn't exist
    if not os.path.exists("uploads"):
        os.makedirs("uploads")

    # generate id
    file_id: str = uuid.uuid4().hex[:8]

    # generate random file name if not provided
    if not file.filename:
        file.filename = f"file_{uuid.uuid4().hex[:8]}.pdf"

    # convert file name to underscore_separated file path
    file_name = file.filename.replace(" ", "_").lower()

    # upload file to uploads directory
    file_location = f"uploads/{file_id}_{file_name}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # save file metadata to database
    new_file = File(id=file_id, name=file.filename, url=file_location)
    session.add(new_file)
    session.commit()
    session.refresh(new_file)

    # documents = parse_pdf_to_documents(new_file)
    ingest_file(new_file)

    # retrieve relevant documents from vector store
    # retriever = Retriever(vector_store, embedding_manager)

    return {
        "filename": file.filename,
        "message": f"File saved at {file_location}",
        "file": new_file,
    }


@app.post("/ask")
def ask(request: AskRequest):
    file_id = request.file_id
    question = request.question

    # process the question
    answer = simple_rag(question, file_id)

    return {
        "question": question,
        "file_id": file_id,
        "message": "Answering your question...",
        "answer": answer,
    }
