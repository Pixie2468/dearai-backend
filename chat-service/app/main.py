import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import datetime, timedelta

from . import models, schemas, auth
from .database import engine, get_db, SessionLocal

logger = logging.getLogger(__name__)

async def cleanup_old_chats_loop():
    while True:
        try:
            cutoff_date = datetime.utcnow() - timedelta(days=30)
            with SessionLocal() as db:
                deleted_count = db.query(models.ChatMessage).filter(
                    models.ChatMessage.created_at < cutoff_date
                ).delete()
                db.commit()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old chat messages.")
        except Exception as e:
            logger.error(f"Error cleaning up old chats: {e}")
            
        # Wait 24 hours before next cleanup
        await asyncio.sleep(86400)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cleanup_task = asyncio.create_task(cleanup_old_chats_loop())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(title="Chat Service", lifespan=lifespan)

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
    after: Optional[datetime] = None,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    query = db.query(models.ChatMessage).filter(models.ChatMessage.user_id == user_id)
    if after:
        query = query.filter(models.ChatMessage.created_at > after)
    chats = query.order_by(models.ChatMessage.created_at.asc()).offset(skip).limit(limit).all()
    return chats
