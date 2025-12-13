import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile
from sqlmodel import select

from app.models import File, SessionDependency

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

    # upload file to uploads directory
    file_location = f"uploads/{file.filename}"
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # generate id
    file_id: str = uuid.uuid4().hex[:8]

    # generate random file name if not provided
    if not file.filename:
        file.filename = f"file_{uuid.uuid4().hex[:8]}.pdf"

    # save file metadata to database
    new_file = File(id=file_id, name=file.filename, url=file_location)
    session.add(new_file)
    session.commit()
    session.refresh(new_file)

    return {
        "filename": file.filename,
        "message": f"File saved at {file_location}",
        "file": new_file,
    }
