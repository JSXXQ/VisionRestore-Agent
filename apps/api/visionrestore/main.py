from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from visionrestore.api.ai_routes import router as ai_router
from visionrestore.api.routes import router
from visionrestore.api.v2_routes import router as v2_router
from visionrestore.core.config import PROJECT_ROOT, get_settings
from visionrestore.schemas.common import ApiError, now_iso
from uuid import uuid4

settings = get_settings()
app = FastAPI(title="VisionRestore Agent", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    if request.url.path.startswith("/api/"):
        return JSONResponse(
            status_code=500,
            content=ApiError(code="INTERNAL_ERROR", message=str(exc), request_id=str(uuid4()), timestamp=now_iso()).model_dump(),
        )
    raise exc

app.include_router(router)
app.include_router(ai_router)
app.include_router(v2_router)

dist = PROJECT_ROOT / "apps" / "web" / "dist"
if dist.exists():
    app.mount("/", StaticFiles(directory=dist, html=True), name="web")