from fastapi import APIRouter

from app.core.dependencies import CurrentUser, DbSession
from app.services.users.schemas import UserResponse, UserUpdate
from app.services.users.service import update_user

router = APIRouter()


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(current_user: CurrentUser):
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_current_user(
    current_user: CurrentUser, db: DbSession, data: UserUpdate
):
    return await update_user(db, current_user.id, data)
