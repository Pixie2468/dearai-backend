from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

from . import models, schemas, auth
from .database import engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Diary Service")

@app.post("/diary", response_model=schemas.DiaryEntryResponse)
def create_diary_entry(
    entry: schemas.DiaryEntryCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    db_entry = models.DiaryEntry(
        user_id=user_id,
        title=entry.title,
        content=entry.content
    )
    db.add(db_entry)
    db.commit()
    db.refresh(db_entry)
    return db_entry

@app.get("/diary", response_model=List[schemas.DiaryEntryResponse])
def get_diary_entries(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    entries = db.query(models.DiaryEntry).filter(models.DiaryEntry.user_id == user_id).order_by(models.DiaryEntry.created_at.desc()).offset(skip).limit(limit).all()
    return entries
