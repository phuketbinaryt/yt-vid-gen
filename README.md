# FramePack Video Generation API

Enterprise-grade REST API for FramePack video generation with job queue management, built with FastAPI and Redis.

## Features

- 🎥 **Image-to-Video Generation**: Convert images to videos with text prompts
- 📝 **Text-to-Video Generation**: Generate videos from text prompts only
- 🔄 **Job Queue System**: Async processing with Redis-backed job management
- 📊 **Real-time Progress Tracking**: Monitor generation progress and status
- 🚀 **Enterprise Ready**: Docker deployment, health checks, API authentication
- 💾 **Memory Optimized**: Supports both high-VRAM and low-VRAM configurations
- 🎛️ **Model Selection**: Choose between standard FramePack and FramePack-F1 models
- 🔧 **GPU Optimized**: Automatic detection and optimization for different GPU types

## Hardware Requirements

### Minimum Requirements
- **GPU**: NVIDIA RTX 30XX/40XX/50XX series with 6GB+ VRAM
- **RAM**: 16GB system RAM
- **Storage**: 50GB+ free space (for models and outputs)
- **OS**: Linux or Windows with Docker support

### Recommended for Production
- **GPU**: NVIDIA H100, A100, or RTX 4090 with 24GB+ VRAM
- **RAM**: 32GB+ system RAM
- **Storage**: 100GB+ NVMe SSD
- **CPU**: 8+ cores for concurrent processing

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd framepack-api

# Copy environment configuration
cp .env.example .env

# Edit configuration as needed
nano .env
```

### 2. Docker Deployment (Recommended)

```bash
# Build and start services
docker-compose up -d

# Check logs
docker-compose logs -f framepack-api

# Check health
curl http://localhost:8000/health
```

### 3. Manual Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

# Start Redis (required)
redis-server

# Start API
python main.py
```

## API Endpoints

### Health Check
```http
GET /health
```

### Video Generation

#### Image-to-Video
```http
POST /api/v1/generate/image-to-video
Content-Type: multipart/form-data

prompt: "The girl dances gracefully, with clear movements, full of charm."
image: [image file]
duration: 5.0
seed: 31337
steps: 25
use_teacache: true
use_f1_model: false
```

#### Text-to-Video
```http
POST /api/v1/generate/text-to-video
Content-Type: application/x-www-form-urlencoded

prompt=A beautiful sunset over mountains
duration=5.0
seed=31337
steps=25
use_teacache=true
use_f1_model=false
```

#### Advanced Generation (JSON)
```http
POST /api/v1/generate
Content-Type: application/json

{
  "prompt": "The girl dances gracefully, with clear movements, full of charm.",
  "mode": "image_to_video",
  "image": "base64_encoded_image_data",
  "duration": 5.0,
  "seed": 31337,
  "steps": 25,
  "cfg_scale": 1.0,
  "distilled_cfg_scale": 10.0,
  "use_teacache": true,
  "use_f1_model": false,
  "gpu_memory_preservation": 6.0,
  "mp4_crf": 16
}
```

### Job Management

#### Check Job Status
```http
GET /api/v1/jobs/{job_id}/status
```

#### Get Job Result
```http
GET /api/v1/jobs/{job_id}/result
```

#### Download Generated Video
```http
GET /api/v1/jobs/{job_id}/download/video
```

#### Cancel Job
```http
DELETE /api/v1/jobs/{job_id}
```

#### List All Jobs
```http
GET /api/v1/jobs?page=1&per_page=50
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_HOST` | `0.0.0.0` | API host address |
| `API_PORT` | `8000` | API port |
| `REDIS_URL` | `redis://localhost:6379/0` | Redis connection URL |
| `GPU_DEVICE` | `cuda:0` | GPU device to use |
| `HIGH_VRAM_THRESHOLD` | `60.0` | VRAM threshold for high-VRAM mode (GB) |
| `MAX_CONCURRENT_JOBS` | `2` | Maximum concurrent video generations |
| `API_KEY` | `None` | Optional API key for authentication |

### GPU Configuration

The API automatically detects available GPU memory and optimizes accordingly:

- **High-VRAM Mode** (>60GB): All models loaded to GPU for maximum speed
- **Low-VRAM Mode** (6-60GB): Dynamic model loading with memory management
- **H100/A100 Support**: Optimized for enterprise GPUs with large VRAM

### Model Selection

- **Standard FramePack**: General-purpose video generation
- **FramePack-F1**: Enhanced model with improved quality and anti-drifting

## Usage Examples

### Python Client

```python
import requests
import base64
import time

# API configuration
API_BASE = "http://localhost:8000"
API_KEY = "your-api-key"  # Optional

headers = {"Authorization": f"Bearer {API_KEY}"} if API_KEY else {}

# Image-to-video generation
with open("input_image.jpg", "rb") as f:
    files = {"image": f}
    data = {
        "prompt": "The girl dances gracefully, with clear movements, full of charm.",
        "duration": 5.0,
        "use_f1_model": True
    }
    
    response = requests.post(
        f"{API_BASE}/api/v1/generate/image-to-video",
        files=files,
        data=data,
        headers=headers
    )
    
    job = response.json()
    job_id = job["job_id"]
    print(f"Job created: {job_id}")

# Monitor progress
while True:
    status_response = requests.get(
        f"{API_BASE}/api/v1/jobs/{job_id}/status",
        headers=headers
    )
    status = status_response.json()
    
    print(f"Status: {status['status']}, Progress: {status['progress']:.1f}%")
    
    if status["status"] == "completed":
        # Download video
        video_response = requests.get(
            f"{API_BASE}/api/v1/jobs/{job_id}/download/video",
            headers=headers
        )
        
        with open(f"generated_video_{job_id}.mp4", "wb") as f:
            f.write(video_response.content)
        
        print("Video downloaded successfully!")
        break
    elif status["status"] == "failed":
        print(f"Generation failed: {status['error_message']}")
        break
    
    time.sleep(5)
```

### cURL Examples

```bash
# Text-to-video generation
curl -X POST "http://localhost:8000/api/v1/generate/text-to-video" \
  -H "Authorization: Bearer your-api-key" \
  -d "prompt=A beautiful sunset over mountains" \
  -d "duration=5.0" \
  -d "use_f1_model=true"

# Check job status
curl "http://localhost:8000/api/v1/jobs/{job_id}/status" \
  -H "Authorization: Bearer your-api-key"

# Download video
curl "http://localhost:8000/api/v1/jobs/{job_id}/download/video" \
  -H "Authorization: Bearer your-api-key" \
  -o generated_video.mp4
```

## Performance Optimization

### For H100/A100 GPUs
```env
HIGH_VRAM_THRESHOLD=60.0
MAX_CONCURRENT_JOBS=4
GPU_MEMORY_PRESERVATION=8.0
```

### For RTX 4090
```env
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2
GPU_MEMORY_PRESERVATION=6.0
```

### For RTX 3070/3080
```env
HIGH_VRAM_THRESHOLD=10.0
MAX_CONCURRENT_JOBS=1
GPU_MEMORY_PRESERVATION=4.0
```

## Monitoring and Logging

### Health Check
```bash
curl http://localhost:8000/health
```

### API Statistics
```bash
curl http://localhost:8000/api/v1/admin/stats
```

### Docker Logs
```bash
docker-compose logs -f framepack-api
```

## Troubleshooting

### Common Issues

1. **Out of Memory Errors**
   - Reduce `MAX_CONCURRENT_JOBS`
   - Increase `GPU_MEMORY_PRESERVATION`
   - Use TeaCache for faster processing

2. **Slow Generation**
   - Enable TeaCache: `use_teacache=true`
   - Reduce steps: `steps=20`
   - Use standard model instead of F1

3. **Model Download Issues**
   - Ensure stable internet connection
   - Check HuggingFace access
   - Verify disk space (30GB+ required)

### Performance Tips

- **Use TeaCache** for 30-50% speed improvement
- **Enable F1 model** for better quality (slower)
- **Adjust steps** based on quality requirements
- **Monitor GPU memory** usage in production

## API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project builds upon FramePack by lllyasviel. Please refer to the original FramePack repository for licensing terms.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review API documentation
3. Check Docker logs for errors
4. Monitor GPU memory usage

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request