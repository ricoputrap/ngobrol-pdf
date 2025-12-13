import uvicorn

from app.models import create_db_and_tables

if __name__ == "__main__":
    create_db_and_tables()
    uvicorn.run("app.app:app", host="0.0.0.0", port=8000, reload=True)
