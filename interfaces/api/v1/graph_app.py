from fastapi import APIRouter


router = APIRouter(prefix="/v1", tags=["graph"])


@router.get("/ping")
def ping() -> dict:
    return {"message": "pong"}


