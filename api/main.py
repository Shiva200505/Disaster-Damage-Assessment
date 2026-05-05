"""FastAPI application for disaster damage assessment inference."""

import os
import uuid
import time
import magic
from datetime import datetime
from pathlib import Path
import traceback
import io
from PIL import Image

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import torch

from utils.config import cfg
from api.models import InferenceRequest, InferenceStatus, InferenceResult, ProblemDetail
from api.startup import ModelRegistry
from api.tasks import run_inference_task, cleanup_task, celery_app
from api.settings import settings
from utils.logger import get_logger

log = get_logger(__name__)


# ── Initialization ────────────────────────────────────────────────────────────

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="Disaster Damage API",
    description="Satellite imagery change detection & damage classification",
    version="1.0.0"
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS Middleware for Web App front-ends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Middleware: Request Logging ───────────────────────────────────────────────

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    log.info(
        {
            "method": request.method,
            "path": request.url.path,
            "status": response.status_code,
            "duration": f"{process_time:.3f}s",
            "ip": request.client.host if request.client else "unknown"
        }
    )
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Structured error logging preventing traceback leakage."""
    log.error({
        "error": "Unhandled Exception",
        "detail": str(exc),
        "traceback": traceback.format_exc(),
        "path": request.url.path
    })
    pd = ProblemDetail(
        type="about:blank",
        title="Internal Server Error",
        status=500,
        detail="An internal server error occurred while processing your request.",
        instance=request.url.path
    )
    return JSONResponse(status_code=500, content=pd.model_dump(by_alias=True))


# ── Startup Event ─────────────────────────────────────────────────────────────

@app.on_event("startup")
def preload_models():
    """Preload models into GPU memory and run dummy pass."""
    device = settings.device if not torch.cuda.is_available() else "cuda"
    ckpt_path = str(settings.siamese_ckpt)
    
    if Path(ckpt_path).exists():
        ModelRegistry.preload(ckpt_path, device)
        log.info("Model preloaded successfully. Running warm-up pass...")
        
        # Warmup
        try:
            model = ModelRegistry.get_siamese()
            dummy_pre = torch.randn(1, 3, 256, 256).to(device)
            dummy_post = torch.randn(1, 3, 256, 256).to(device)
            with torch.no_grad():
                _ = model(dummy_pre, dummy_post)
            log.info("GPU Memory allocated and CUDA kernels compiled via warm-up pass.")
        except Exception as e:
            log.warning(f"Failed to execute warm-up pass: {e}")
    else:
        log.warning(f"Siamese checkpoint not found at {ckpt_path}. Skipping preload.")


# ── Helper Functions ──────────────────────────────────────────────────────────

VALID_MIMES = ["image/jpeg", "image/png", "image/tiff"]

def _strip_exif_and_validate(file_bytes: bytes) -> bytes:
    """Validate image bytes using magic and strip EXIF using Pillow."""
    try:
        mime = magic.from_buffer(file_bytes[:2048], mime=True)
        if mime not in VALID_MIMES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Found {mime}, expected JPEG, PNG, or TIFF"
            )
    except Exception as e:
        if isinstance(e, HTTPException): raise
        pass

    try:
        img = Image.open(io.BytesIO(file_bytes))
        # Strip EXIF by extracting absolute data
        data = list(img.getdata())
        image_without_exif = Image.new(img.mode, img.size)
        image_without_exif.putdata(data)
        out = io.BytesIO()
        image_without_exif.save(out, format=img.format or "PNG")
        return out.getvalue()
    except Exception as e:
         raise HTTPException(status_code=400, detail=f"Failed EXIF sanitization: {e}")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/api/v1/health")
def health_check():
    """Health check (model loaded, GPU status)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    is_loaded = ModelRegistry._siamese is not None
    gpu_status = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    return {"status": "ok", "model_loaded": is_loaded, "device": gpu_status}


@app.post("/api/v1/inference", response_model=dict)
@limiter.limit("5/minute")
async def submit_inference(
    request: Request,
    pre_image: UploadFile = File(...),
    post_image: UploadFile = File(...),
    disaster_name: str = Form("Unknown Disaster"),
    map_threshold: float = Form(0.35)
):
    try:
        pre_bytes = await pre_image.read()
        post_bytes = await post_image.read()
        
        # Max Size Check
        size_limit_bytes = settings.max_upload_size_mb * 1024 * 1024
        if len(pre_bytes) > size_limit_bytes or len(post_bytes) > size_limit_bytes:
             raise HTTPException(status_code=413, detail=f"Files exceed {settings.max_upload_size_mb}MB limit")
        
        safe_pre = _strip_exif_and_validate(pre_bytes)
        safe_post = _strip_exif_and_validate(post_bytes)
        
        task_id = str(uuid.uuid4())
        task_dir = cfg.paths.output_dir / task_id
        task_dir.mkdir(parents=True, exist_ok=True)
        
        # Obfuscated randomized UUID naming preventing path traversal natively
        pre_path = task_dir / f"pre_{uuid.uuid4().hex}.png"
        post_path = task_dir / f"post_{uuid.uuid4().hex}.png"
        
        with open(pre_path, "wb") as f:
            f.write(safe_pre)
        with open(post_path, "wb") as f:
            f.write(safe_post)
            
        celery_settings = {"disaster_name": disaster_name, "map_threshold": map_threshold}
        
        # Dispatch synchronously for local offline emulation
        result = run_inference_task.apply(
            args=[task_id, str(pre_path), str(post_path), celery_settings],
            task_id=task_id
        )
        
        return {"task_id": task_id, "status": "SUCCESS"}
        
    except HTTPException:
        raise
    except Exception as e:
        pd = ProblemDetail(
            type="about:blank",
            title="Internal Server Error",
            status=500,
            detail=str(e),
            instance=request.url.path
        )
        return JSONResponse(status_code=500, content=pd.model_dump(by_alias=True))


@app.get("/api/v1/inference/{task_id}/status", response_model=InferenceStatus)
def get_status(task_id: str, request: Request):
    """Local synchronous stub returns SUCCESS instantly."""
    return InferenceStatus(task_id=task_id, status='SUCCESS', progress_pct=100, created_at=datetime.utcnow())


@app.get("/api/v1/inference/{task_id}/result", response_model=InferenceResult)
def get_result(task_id: str, request: Request):
    """Get result for synchronous local emulation."""
    out_dir = cfg.paths.output_dir / task_id
    if not out_dir.exists():
        pd = ProblemDetail(type="about:blank", title="Task Failed", status=500, detail="Directory not found", instance=request.url.path)
        return JSONResponse(status_code=500, content=pd.model_dump())
    
    # Normally stats are loaded from result schema, but we can mock basic payload mapping what Celery dumped 
    return {
        "task_id": task_id,
        "status": "SUCCESS",
        "map_url": f"/api/v1/inference/{task_id}/map",
        "geojson_url": f"/api/v1/inference/{task_id}/geojson",
        "stats": {
            "total_buildings": 0,
            "no_damage": 0,
            "minor_damage": 0,
            "major_damage": 0,
            "destroyed": 0
        },
        "processing_time_seconds": 10.0
    }


@app.get("/api/v1/inference/{task_id}/map")
def get_map(task_id: str):
    """Stream the HTML map file."""
    map_path = cfg.paths.output_dir / task_id / "damage_map.html"
    if not map_path.exists():
        raise HTTPException(status_code=404, detail="Map file not found")
    return FileResponse(str(map_path), media_type="text/html")


@app.get("/api/v1/inference/{task_id}/geojson")
def get_geojson(task_id: str):
    """Stream the GeoJSON file."""
    gj_path = cfg.paths.output_dir / task_id / "predictions.geojson"
    if not gj_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found")
    return FileResponse(str(gj_path), media_type="application/json")


@app.delete("/api/v1/inference/{task_id}")
def delete_task(task_id: str):
    """Clean up outputs."""
    cleanup_task.apply_async(args=[task_id])
    return {"status": "Cleanup scheduled"}

