from pydantic import BaseModel, Field, validator
from typing import Optional, Union, Literal, List
from enum import Enum
import base64

class JobStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class GenerationMode(str, Enum):
    IMAGE_TO_VIDEO = "image_to_video"
    TEXT_TO_VIDEO = "text_to_video"

class VideoGenerationRequest(BaseModel):
    # Core parameters
    prompt: str = Field(..., description="Text prompt describing the desired video content")
    mode: GenerationMode = Field(default=GenerationMode.IMAGE_TO_VIDEO, description="Generation mode")
    
    # Image input (optional for text-to-video)
    image: Optional[str] = Field(None, description="Base64 encoded image for image-to-video generation")
    image_url: Optional[str] = Field(None, description="URL to image for image-to-video generation")
    
    # Video parameters
    duration: float = Field(default=5.0, ge=0.1, le=120.0, description="Video duration in seconds")
    aspect_ratio: Optional[str] = Field(None, description="Video aspect ratio (e.g., '16:9', '9:16', '1:1', '4:3', '21:9')")
    seed: int = Field(default=31337, description="Random seed for reproducible results")
    
    # Advanced parameters
    steps: int = Field(default=25, ge=1, le=100, description="Number of diffusion steps")
    cfg_scale: float = Field(default=1.0, ge=1.0, le=32.0, description="CFG scale")
    distilled_cfg_scale: float = Field(default=10.0, ge=1.0, le=32.0, description="Distilled CFG scale")
    cfg_rescale: float = Field(default=0.0, ge=0.0, le=1.0, description="CFG rescale factor")
    
    # Performance options
    use_teacache: bool = Field(default=True, description="Use TeaCache for faster generation")
    gpu_memory_preservation: float = Field(default=6.0, ge=6.0, le=128.0, description="GPU memory to preserve (GB)")
    
    # Output options
    mp4_crf: int = Field(default=16, ge=0, le=100, description="MP4 compression quality (lower = better)")
    
    # Model selection
    use_f1_model: bool = Field(default=False, description="Use FramePack-F1 model instead of standard")
    
    @validator('aspect_ratio')
    def validate_aspect_ratio(cls, v):
        if v is None:
            return v
        
        # Supported aspect ratios
        supported_ratios = {
            "16:9", "9:16", "1:1", "4:3", "3:2", "21:9", "2.35:1", "2:3", "3:4"
        }
        
        if v in supported_ratios:
            return v
        
        # Try to parse custom ratio like "16:9"
        try:
            parts = v.split(':')
            if len(parts) == 2:
                w_ratio = float(parts[0])
                h_ratio = float(parts[1])
                if w_ratio > 0 and h_ratio > 0:
                    return v
        except (ValueError, IndexError):
            pass
        
        raise ValueError(f"Invalid aspect ratio '{v}'. Supported ratios: {', '.join(sorted(supported_ratios))} or custom format like '16:9'")
    
    @validator('image_url')
    def validate_image_inputs(cls, v, values):
        mode = values.get('mode')
        image = values.get('image')
        
        if mode == GenerationMode.IMAGE_TO_VIDEO:
            if not image and not v:
                raise ValueError("Either 'image' (base64) or 'image_url' is required for image-to-video generation")
            if image and v:
                raise ValueError("Provide either 'image' (base64) or 'image_url', not both")
        
        if image:
            try:
                # Validate base64 format
                base64.b64decode(image)
            except Exception:
                raise ValueError("Invalid base64 image format")
        
        if v:
            # Basic URL validation
            if not v.startswith(('http://', 'https://')):
                raise ValueError("Image URL must start with http:// or https://")
        
        return v

class JobResponse(BaseModel):
    job_id: str
    status: JobStatus
    message: str
    created_at: str

class JobStatusResponse(BaseModel):
    job_id: str
    status: JobStatus
    progress: float = Field(ge=0.0, le=100.0)
    message: str
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error_message: Optional[str] = None
    estimated_time_remaining: Optional[int] = None  # seconds
    current_step: Optional[int] = None
    total_steps: Optional[int] = None
    generated_frames: Optional[int] = None
    video_length: Optional[float] = None  # seconds

class JobResultResponse(BaseModel):
    job_id: str
    status: JobStatus
    video_url: Optional[str] = None
    preview_url: Optional[str] = None
    metadata: Optional[dict] = None
    created_at: str
    completed_at: Optional[str] = None
    generation_time: Optional[float] = None  # seconds

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    job_id: Optional[str] = None

class HealthResponse(BaseModel):
    status: str
    gpu_available: bool
    gpu_memory_free: Optional[float] = None  # GB
    gpu_memory_total: Optional[float] = None  # GB
    active_jobs: int
    queue_size: int
    models_loaded: bool

class JobListResponse(BaseModel):
    jobs: List[JobStatusResponse]
    total: int
    page: int
    per_page: int