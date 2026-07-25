import asyncio
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
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
                deleted_count = db.query(models.ChatSession).filter(
                    models.ChatSession.updated_at < cutoff_date
                ).delete()
                db.commit()
                if deleted_count > 0:
                    logger.info(f"Cleaned up {deleted_count} old chat sessions.")
        except Exception as e:
            logger.error(f"Error cleaning up old chats: {e}")
            
        # Wait 24 hours before next cleanup
        await asyncio.sleep(86400)

@asynccontextmanager
async def lifespan(app: FastAPI):
    models.Base.metadata.create_all(bind=engine)
    cleanup_task = asyncio.create_task(cleanup_old_chats_loop())
    yield
    cleanup_task.cancel()
    try:
        await cleanup_task
    except asyncio.CancelledError:
        pass

app = FastAPI(title="Chat Service", lifespan=lifespan)

@app.post("/sessions", response_model=schemas.ChatSessionResponse)
def create_session(
    session_data: schemas.ChatSessionBase,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    db_session = models.ChatSession(
        user_id=user_id,
        title=session_data.title
    )
    db.add(db_session)
    db.commit()
    db.refresh(db_session)
    return db_session

@app.get("/sessions", response_model=List[schemas.ChatSessionResponse])
def get_sessions(
    skip: int = 0,
    limit: int = 50,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    return db.query(models.ChatSession).filter(models.ChatSession.user_id == user_id).order_by(models.ChatSession.updated_at.desc()).offset(skip).limit(limit).all()

@app.patch("/sessions/{session_id}", response_model=schemas.ChatSessionResponse)
def update_session(
    session_id: str,
    session_update: schemas.ChatSessionUpdate,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == session_id, models.ChatSession.user_id == user_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    db_session.title = session_update.title
    db_session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_session)
    return db_session

@app.post("/chats", response_model=schemas.ChatMessageResponse)
def create_chat(
    chat: schemas.ChatMessageCreate,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    db_session = db.query(models.ChatSession).filter(models.ChatSession.id == chat.session_id, models.ChatSession.user_id == user_id).first()
    if not db_session:
        raise HTTPException(status_code=404, detail="Session not found")

    db_chat = models.ChatMessage(
        user_id=user_id,
        session_id=chat.session_id,
        role=chat.role,
        content=chat.content
    )
    db.add(db_chat)
    db_session.updated_at = datetime.utcnow()
    db.commit()
    db.refresh(db_chat)
    return db_chat

@app.get("/chats", response_model=List[schemas.ChatMessageResponse])
def get_chats(
    session_id: str,
    skip: int = 0,
    limit: int = 100,
    after: Optional[datetime] = None,
    db: Session = Depends(get_db),
    user_id: str = Depends(auth.verify_internal_token)
):
    query = db.query(models.ChatMessage).filter(
        models.ChatMessage.user_id == user_id,
        models.ChatMessage.session_id == session_id
    )
    if after:
        query = query.filter(models.ChatMessage.created_at > after)
    chats = query.order_by(models.ChatMessage.created_at.asc()).offset(skip).limit(limit).all()
    return chats
