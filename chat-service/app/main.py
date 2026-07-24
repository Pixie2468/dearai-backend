from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List

from . import models, schemas, auth
from .database import engine, get_db

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chat Service")

@app.post("/chats", response_model=schemas.ChatMessageResponse)
def create_chat(
    chat: schemas.ChatMessageCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    db_chat = models.ChatMessage(
        user_id=user_id,
        role=chat.role,
        content=chat.content
    )
    db.add(db_chat)
    db.commit()
    db.refresh(db_chat)
    return db_chat

@app.get("/chats", response_model=List[schemas.ChatMessageResponse])
def get_chats(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    chats = db.query(models.ChatMessage).filter(models.ChatMessage.user_id == user_id).order_by(models.ChatMessage.created_at.asc()).offset(skip).limit(limit).all()
    return chats
