from fastapi import FastAPI

try:
    from .v1.graph_app import router as v1_router
except Exception:
    v1_router = None


app = FastAPI(title="KRag API", version="0.1.0")

if v1_router is not None:
    app.include_router(v1_router)


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok"}


