# FramePack API - Clean Ubuntu Deployment Guide

This guide provides instructions for deploying the FramePack API on a clean Ubuntu server, following the official FramePack installation requirements.

## 🎯 **Recommended Setup**

### **RunPod Configuration**
- **Template**: Ubuntu 22.04 LTS (clean, no pre-installed ML libraries)
- **GPU**: RTX 4090 (24GB VRAM) - $0.34/hr
- **Storage**: 50GB+ SSD
- **RAM**: 32GB+ recommended

### **Alternative Platforms**
- **AWS EC2**: g5.xlarge or g5.2xlarge with Ubuntu 22.04
- **Google Cloud**: n1-standard-4 with T4/V100 GPU
- **Azure**: NC6s_v3 with Ubuntu 22.04
- **Local**: Ubuntu 22.04 with NVIDIA GPU

## 🚀 **One-Command Deployment**

### **RunPod Startup Command**
```bash
bash <(curl -s https://raw.githubusercontent.com/phuketbinaryt/yt-vid-gen/main/runpod-startup.sh)
```

### **Manual Installation**
```bash
# Clone the repository
git clone https://github.com/phuketbinaryt/yt-vid-gen.git
cd yt-vid-gen

# Run the startup script
chmod +x runpod-startup.sh
./runpod-startup.sh
```

## 📋 **What the Script Does**

### **1. System Setup**
- Updates Ubuntu packages
- Installs build tools and Python development headers
- Sets up CUDA toolkit for GPU support
- Installs Redis server

### **2. FramePack Installation**
- Clones official FramePack repository
- Installs PyTorch with CUDA support using official index
- Installs all FramePack dependencies from requirements.txt
- Adds attention optimizations (xformers, flash-attn)

### **3. API Setup**
- Installs FastAPI and related web framework dependencies
- Sets up Redis for job queue management
- Configures environment variables
- Creates necessary directories

### **4. Verification**
- Tests all imports (PyTorch, diffusers, transformers, FramePack modules)
- Verifies CUDA availability
- Confirms FramePack integration

### **5. Service Startup**
- Starts Redis server
- Launches Celery worker for background processing
- Starts FastAPI server on port 8000

## 🔧 **Configuration**

### **Environment Variables**
The script automatically creates a `.env` file with optimal settings:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379/0

# GPU Settings
GPU_DEVICE=cuda:0
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2

# Model Paths (HuggingFace)
HUNYUAN_MODEL_PATH=hunyuanvideo-community/HunyuanVideo
FLUX_REDUX_MODEL_PATH=black-forest-labs/FLUX.1-Redux-dev
FRAMEPACK_MODEL_PATH=lllyasviel/FramePack
FRAMEPACK_F1_MODEL_PATH=lllyasviel/FramePack-F1

# Storage Directories
UPLOAD_DIR=/workspace/uploads
OUTPUT_DIR=/workspace/outputs
TEMP_DIR=/workspace/temp
```

### **Customization**
You can modify the `.env` file after installation to:
- Change API port
- Adjust GPU memory settings
- Set API key for authentication
- Modify model paths
- Change storage locations

## 🎮 **Usage**

### **API Endpoints**
Once deployed, the API will be available at:
- **Base URL**: `http://your-server:8000`
- **Documentation**: `http://your-server:8000/docs`
- **Health Check**: `http://your-server:8000/health`

### **Video Generation**
```bash
# Image-to-video generation
curl -X POST "http://your-server:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over the ocean",
    "mode": "image_to_video",
    "image": "base64_encoded_image_here",
    "duration": 5,
    "steps": 30
  }'

# Text-to-video generation
curl -X POST "http://your-server:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A cat playing in a garden",
    "mode": "text_to_video",
    "duration": 5,
    "steps": 30
  }'
```

### **Job Management**
```bash
# Check job status
curl "http://your-server:8000/jobs/{job_id}/status"

# Download generated video
curl "http://your-server:8000/jobs/{job_id}/download" -o video.mp4
```

## 🔍 **Monitoring**

### **Logs**
```bash
# API logs
tail -f /workspace/yt-vid-gen/api.log

# Celery worker logs
tail -f /workspace/yt-vid-gen/celery.log

# System logs
journalctl -u redis-server -f
```

### **GPU Usage**
```bash
# Monitor GPU usage
nvidia-smi -l 1

# Check CUDA availability
python3 -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

## 🛠 **Troubleshooting**

### **Common Issues**

#### **CUDA Not Available**
```bash
# Check NVIDIA driver
nvidia-smi

# Reinstall CUDA toolkit
apt install -y cuda-toolkit-12-4
export PATH=/usr/local/cuda/bin:$PATH
```

#### **Import Errors**
```bash
# Check Python path
echo $PYTHONPATH

# Verify FramePack installation
cd /workspace/yt-vid-gen/FramePack
python3 -c "from diffusers_helper.hunyuan import encode_prompt_conds; print('OK')"
```

#### **Redis Connection Issues**
```bash
# Check Redis status
redis-cli ping

# Restart Redis
systemctl restart redis-server
```

#### **Out of Memory**
```bash
# Check GPU memory
nvidia-smi

# Reduce concurrent jobs in .env
MAX_CONCURRENT_JOBS=1
GPU_MEMORY_PRESERVATION=8.0
```

## 📊 **Performance**

### **Expected Performance on RTX 4090**
- **5-second video**: ~2-3 minutes generation time
- **Memory usage**: ~18-20GB VRAM
- **Concurrent jobs**: 1-2 depending on video length
- **Quality**: High-quality 640x640 or higher resolution

### **Optimization Tips**
- Use F1 model for faster generation: `"use_f1_model": true`
- Enable TeaCache for speed: `"use_teacache": true`
- Adjust steps for quality vs speed trade-off
- Monitor GPU memory and adjust preservation settings

## 🔒 **Security**

### **API Key Authentication**
Uncomment and set in `.env`:
```env
API_KEY=your-secure-api-key-here
```

### **Firewall**
```bash
# Allow API port
ufw allow 8000

# Restrict Redis access
ufw deny 6379
```

## 📈 **Scaling**

### **Multiple Workers**
```bash
# Start additional Celery workers
celery -A framepack_worker worker --loglevel=info --concurrency=1 --hostname=worker2@%h &
```

### **Load Balancing**
Use nginx or similar to distribute requests across multiple API instances.

## 🆘 **Support**

For issues and questions:
- Check the logs first
- Verify GPU and CUDA setup
- Ensure all dependencies are installed correctly
- Test with simple requests before complex ones

This clean installation approach eliminates dependency conflicts and provides a stable, production-ready FramePack API deployment.