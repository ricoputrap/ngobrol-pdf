from pydantic import BaseModel


class AskRequest(BaseModel):
    file_id: str
    question: str
