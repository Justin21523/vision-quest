# FastAPI main app - see phase1_fastapi_main artifact
"""
FastAPI main application for VisionQuest.
Provides multimodal AI capabilities with RESTful API.
"""

import time
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.api.v1 import health, caption, vqa, chat, models, rag, history, agent, game

# Configure logging
logging.basicConfig(
    level=logging.INFO if not settings.DEBUG else logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


from app.database.init_db import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    # Startup
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")

    # Initialize database tables
    try:
        init_db()
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

    yield

    # Shutdown
    logger.info("Shutting down VisionQuest application")
    # TODO: Cleanup model resources


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A portable multimodal AI assistant for vision and text understanding",
    lifespan=lifespan,
    openapi_url=f"{settings.API_PREFIX}/openapi.json",
    docs_url=f"{settings.API_PREFIX}/docs",
    redoc_url=f"{settings.API_PREFIX}/redoc",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "status_code": exc.status_code,
            "path": str(request.url.path),
        },
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "status_code": 500,
            "path": str(request.url.path),
        },
    )


# Include API routes
app.include_router(health.router, prefix=settings.API_PREFIX, tags=["health"])
app.include_router(caption.router, prefix=settings.API_PREFIX)
app.include_router(vqa.router, prefix=settings.API_PREFIX)
app.include_router(chat.router, prefix=settings.API_PREFIX)
app.include_router(models.router, prefix=f"{settings.API_PREFIX}/models", tags=["models"])
app.include_router(rag.router, prefix=f"{settings.API_PREFIX}/rag", tags=["rag"])
app.include_router(history.router, prefix=f"{settings.API_PREFIX}/history", tags=["history"])
app.include_router(agent.router, prefix=f"{settings.API_PREFIX}/agent", tags=["agent"])
app.include_router(game.router, prefix=f"{settings.API_PREFIX}/game", tags=["game"])

# TODO: Add other routers in subsequent phases
# app.include_router(caption.router, prefix=settings.API_PREFIX, tags=["caption"])
# app.include_router(vqa.router, prefix=settings.API_PREFIX, tags=["vqa"])
# app.include_router(chat.router, prefix=settings.API_PREFIX, tags=["chat"])
# app.include_router(rag.router, prefix=settings.API_PREFIX, tags=["rag"])
# app.include_router(agent.router, prefix=settings.API_PREFIX, tags=["agent"])
# app.include_router(game.router, prefix=settings.API_PREFIX, tags=["game"])


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint with application information."""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "environment": settings.ENV,
        "docs_url": f"{settings.API_PREFIX}/docs",
        "health_url": f"{settings.API_PREFIX}/health",
    }


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
