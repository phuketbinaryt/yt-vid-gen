#!/bin/bash

# RunPod startup script for FramePack API
echo "🚀 Starting FramePod API deployment on RunPod..."

# Update system
apt update && apt install -y redis-server curl

# Clone repository if not exists
if [ ! -d "/workspace/yt-vid-gen" ]; then
    cd /workspace
    git clone https://github.com/phuketbinaryt/yt-vid-gen.git
    cd yt-vid-gen
else
    cd /workspace/yt-vid-gen
    git pull origin main
fi

# Clone FramePack repository
echo "📥 Cloning FramePack repository..."
if [ ! -d "FramePack" ]; then
    git clone https://github.com/lllyasviel/FramePack.git
    cd FramePack
    git checkout main
    cd ..
else
    echo "FramePack already exists, updating..."
    cd FramePack
    git pull origin main
    cd ..
fi

# Install Python dependencies
echo "📦 Installing Python dependencies..."
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ requirements.txt not found, installing basic dependencies..."
    pip install fastapi uvicorn redis celery pillow requests python-multipart pydantic-settings
fi

# Uninstall conflicting packages first
pip uninstall -y gradio torch torchvision torchaudio xformers flash-attn

# Install PyTorch 2.6.0 with CUDA 12.4 (latest available, meets FramePack 2.6+ requirement)
pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 torchaudio==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install latest xformers compatible with PyTorch 2.6
pip install xformers --index-url https://download.pytorch.org/whl/cu124

# Install flash-attn (should work better with PyTorch 2.6)
pip install flash-attn --no-build-isolation

# Install FramePack dependencies
echo "📦 Installing FramePack dependencies..."
cd FramePack
pip install -r requirements.txt
cd ..

# Setup environment
cp .env.example .env

# Configure for RunPod
cat > .env << EOF
# RunPod Configuration
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379/0

# GPU Configuration (auto-detect)
GPU_DEVICE=cuda:0
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2
GPU_MEMORY_PRESERVATION=6.0

# File Storage
UPLOAD_DIR=/workspace/uploads
OUTPUT_DIR=/workspace/outputs
TEMP_DIR=/workspace/temp

# Optional API Key (uncomment to enable)
# API_KEY=your-secret-key-here
EOF

# Create directories
mkdir -p /workspace/uploads /workspace/outputs /workspace/temp

# Start Redis
redis-server --daemonize yes --bind 0.0.0.0

# Wait for Redis to start
sleep 2

# Verify FramePack installation
echo "🔍 Verifying FramePack installation..."
ls -la /workspace/yt-vid-gen/FramePack/
echo "📁 FramePack contents:"
ls -la /workspace/yt-vid-gen/FramePack/diffusers_helper/ || echo "diffusers_helper directory not found"

# Set PYTHONPATH to include FramePack
export PYTHONPATH="/workspace/yt-vid-gen/FramePack:$PYTHONPATH"
echo "🐍 PYTHONPATH set to: $PYTHONPATH"

# Set PyTorch memory management for better GPU memory handling
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "🔧 PyTorch CUDA memory configuration set"

# Additional CUDA environment variables for RTX 5090 compatibility
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="5.0;6.0;7.0;7.5;8.0;8.6;9.0;12.0"
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
echo "🚀 CUDA environment optimized for RTX 5090"

# Start the API
echo "🎬 Starting FramePack API on port 8000..."
cd /workspace/yt-vid-gen
python main.py &

# Keep container running
echo "✅ FramePack API is running!"
echo "📡 API URL: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"

# Show logs
tail -f /dev/null