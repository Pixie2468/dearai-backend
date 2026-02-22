from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.models import User
from app.services.users.schemas import UserUpdate


async def get_user_by_id(db: AsyncSession, user_id: UUID) -> User | None:
    result = await db.execute(select(User).where(User.id == user_id))
    return result.scalar_one_or_none()


async def get_user_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalar_one_or_none()


async def update_user(db: AsyncSession, user_id: UUID, data: UserUpdate) -> User | None:
    update_data = data.model_dump(exclude_unset=True)
    if not update_data:
        return await get_user_by_id(db, user_id)

    await db.execute(update(User).where(User.id == user_id).values(**update_data))
    return await get_user_by_id(db, user_id)
