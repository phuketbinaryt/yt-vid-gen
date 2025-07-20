from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, File, UploadFile, Query
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
import torch
import base64
from typing import Optional, List
from datetime import datetime
import asyncio
import threading
import time
from contextlib import asynccontextmanager

from config import settings
from models import (
    VideoGenerationRequest, JobResponse, JobStatusResponse, JobResultResponse, 
    ErrorResponse, HealthResponse, JobListResponse, JobStatus, GenerationMode
)
from job_manager import job_manager
from framepack_worker import framepack_worker

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global worker_thread, worker_running
    
    # Startup
    print("Starting FramePack API...")
    
    # Start background worker
    worker_thread = threading.Thread(target=background_worker, daemon=True)
    worker_thread.start()
    
    print("FramePack API started successfully!")
    
    yield
    
    # Shutdown
    worker_running = False
    print("Shutting down FramePack API...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="FramePack Video Generation API",
    description="Enterprise-grade API for FramePack video generation with job queue management",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for outputs
app.mount("/outputs", StaticFiles(directory=settings.OUTPUT_DIR), name="outputs")
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")

# Security
security = HTTPBearer(auto_error=False)

def verify_api_key(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify API key if configured"""
    if settings.API_KEY:
        if not credentials or credentials.credentials != settings.API_KEY:
            raise HTTPException(status_code=401, detail="Invalid API key")
    return True

# Background worker thread
worker_thread = None
worker_running = False

def background_worker():
    """Background worker to process jobs"""
    global worker_running
    worker_running = True
    
    while worker_running:
        try:
            # Get next job
            job_id = job_manager.get_next_job()
            
            if job_id:
                print(f"Processing job: {job_id}")
                framepack_worker.process_job(job_id)
            else:
                # No jobs, sleep briefly
                time.sleep(1)
                
        except Exception as e:
            print(f"Worker error: {e}")
            time.sleep(5)  # Wait before retrying


# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check API health and GPU status"""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_memory_free = None
        gpu_memory_total = None
        
        if gpu_available:
            gpu_memory_free = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        return HealthResponse(
            status="healthy",
            gpu_available=gpu_available,
            gpu_memory_free=gpu_memory_free,
            gpu_memory_total=gpu_memory_total,
            active_jobs=job_manager.get_active_jobs_count(),
            queue_size=job_manager.get_queue_size(),
            models_loaded=framepack_worker.models_loaded
        )
    except Exception as e:
        return HealthResponse(
            status="unhealthy",
            gpu_available=False,
            active_jobs=0,
            queue_size=0,
            models_loaded=False
        )

# Video generation endpoints
@app.post("/api/v1/generate", response_model=JobResponse)
async def generate_video(
    request: VideoGenerationRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate a video from image and/or text prompt"""
    try:
        # Validate request
        if request.mode == GenerationMode.IMAGE_TO_VIDEO and not request.image:
            raise HTTPException(status_code=400, detail="Image is required for image-to-video generation")
        
        # Create job
        job_data = request.dict()
        job_id = job_manager.create_job(job_data)
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Job created and queued for processing",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/image-to-video", response_model=JobResponse)
async def generate_image_to_video(
    request: VideoGenerationRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate video from image (base64 or URL) and text prompt"""
    try:
        # Ensure it's image-to-video mode
        request.mode = GenerationMode.IMAGE_TO_VIDEO
        
        # Validate that either image or image_url is provided
        if not request.image and not request.image_url:
            raise HTTPException(status_code=400, detail="Either 'image' (base64) or 'image_url' is required")
        
        job_id = job_manager.create_job(request.dict())
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Image-to-video job created and queued",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/text-to-video", response_model=JobResponse)
async def generate_text_to_video(
    request: VideoGenerationRequest,
    _: bool = Depends(verify_api_key)
):
    """Generate video from text prompt only"""
    try:
        # Ensure it's text-to-video mode
        request.mode = GenerationMode.TEXT_TO_VIDEO
        
        job_id = job_manager.create_job(request.dict())
        
        return JobResponse(
            job_id=job_id,
            status=JobStatus.PENDING,
            message="Text-to-video job created and queued",
            created_at=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Job management endpoints
@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Get job status and progress"""
    job_status = job_manager.get_job_status_response(job_id)
    if not job_status:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return job_status

@app.get("/api/v1/jobs/{job_id}/result", response_model=JobResultResponse)
async def get_job_result(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Get job result with download URLs"""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result_data = job_data.get("result_data", {})
    
    # Generate static file URLs
    video_url = None
    preview_url = None
    
    if result_data.get("video_filename"):
        # Extract filename from full path
        video_filename = os.path.basename(result_data["video_filename"])
        video_url = f"/outputs/{video_filename}"
    
    if result_data.get("input_filename"):
        # Extract filename from full path
        input_filename = os.path.basename(result_data["input_filename"])
        preview_url = f"/outputs/{input_filename}"
    
    return JobResultResponse(
        job_id=job_id,
        status=job_data["status"],
        video_url=video_url,
        preview_url=preview_url,
        metadata=result_data,
        created_at=job_data["created_at"],
        completed_at=job_data.get("completed_at"),
        generation_time=None  # Calculate if needed
    )

@app.get("/api/v1/jobs/{job_id}/download/video")
async def download_video(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Download generated video"""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job_data["status"] != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job not completed yet")
    
    result_data = job_data.get("result_data", {})
    video_filename = result_data.get("video_filename")
    
    if not video_filename or not os.path.exists(video_filename):
        raise HTTPException(status_code=404, detail="Video file not found")
    
    return FileResponse(
        video_filename,
        media_type="video/mp4",
        filename=f"{job_id}_generated_video.mp4"
    )

@app.get("/api/v1/jobs/{job_id}/download/input")
async def download_input_image(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Download input image"""
    job_data = job_manager.get_job(job_id)
    if not job_data:
        raise HTTPException(status_code=404, detail="Job not found")
    
    result_data = job_data.get("result_data", {})
    input_filename = result_data.get("input_filename")
    
    if not input_filename or not os.path.exists(input_filename):
        raise HTTPException(status_code=404, detail="Input image not found")
    
    return FileResponse(
        input_filename,
        media_type="image/png",
        filename=f"{job_id}_input_image.png"
    )

@app.delete("/api/v1/jobs/{job_id}")
async def cancel_job(
    job_id: str,
    _: bool = Depends(verify_api_key)
):
    """Cancel a job"""
    success = job_manager.cancel_job(job_id)
    if not success:
        raise HTTPException(status_code=400, detail="Job cannot be cancelled")
    
    return {"message": "Job cancelled successfully"}

@app.get("/api/v1/jobs", response_model=JobListResponse)
async def list_jobs(
    page: int = Query(default=1, ge=1),
    per_page: int = Query(default=50, ge=1, le=100),
    _: bool = Depends(verify_api_key)
):
    """List all jobs with pagination"""
    jobs_data, total = job_manager.get_all_jobs(page, per_page)
    
    jobs = []
    for job_data in jobs_data:
        jobs.append(JobStatusResponse(
            job_id=job_data["job_id"],
            status=job_data["status"],
            progress=job_data["progress"],
            message=job_data["message"],
            created_at=job_data["created_at"],
            started_at=job_data.get("started_at"),
            completed_at=job_data.get("completed_at"),
            error_message=job_data.get("error_message"),
            estimated_time_remaining=job_data.get("estimated_time_remaining"),
            current_step=job_data.get("current_step"),
            total_steps=job_data.get("total_steps"),
            generated_frames=job_data.get("generated_frames"),
            video_length=job_data.get("video_length")
        ))
    
    return JobListResponse(
        jobs=jobs,
        total=total,
        page=page,
        per_page=per_page
    )

# Admin endpoints
@app.post("/api/v1/admin/cleanup")
async def cleanup_old_jobs(_: bool = Depends(verify_api_key)):
    """Clean up old completed jobs"""
    job_manager.cleanup_old_jobs()
    return {"message": "Cleanup completed"}

@app.get("/api/v1/admin/stats")
async def get_stats(_: bool = Depends(verify_api_key)):
    """Get API statistics"""
    return {
        "active_jobs": job_manager.get_active_jobs_count(),
        "queue_size": job_manager.get_queue_size(),
        "models_loaded": framepack_worker.models_loaded,
        "high_vram_mode": framepack_worker.high_vram,
        "gpu_available": torch.cuda.is_available()
    }

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(error=exc.detail).dict()
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            detail=str(exc)
        ).dict()
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        workers=settings.API_WORKERS,
        reload=settings.DEBUG
    )