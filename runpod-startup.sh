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
echo "🧹 Cleaning up conflicting packages..."
pip uninstall -y gradio torch torchvision torchaudio xformers flash-attn

# Install PyTorch 2.6.0 with exact version matching
echo "🔥 Installing PyTorch 2.6.0 with exact version matching..."
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --force-reinstall

# Install exact flash-attn version compatible with xformers (2.7.1-2.8.0)
echo "💫 Installing exact flash-attn version for xformers compatibility..."
FLASH_ATTN_VERSIONS=("2.8.0" "2.7.1")
FLASH_ATTN_INSTALLED=false

for version in "${FLASH_ATTN_VERSIONS[@]}"; do
    echo "🔄 Trying flash-attn==$version..."
    if pip install flash-attn==$version --no-build-isolation --force-reinstall; then
        echo "✅ flash-attn $version installed successfully"
        FLASH_ATTN_INSTALLED=true
        break
    else
        echo "❌ flash-attn $version failed"
    fi
done

if [ "$FLASH_ATTN_INSTALLED" = false ]; then
    echo "❌ All flash-attn versions failed"
    echo "🔄 Continuing without flash-attn (will use standard attention)"
    export DISABLE_FLASH_ATTN=1
fi

# Install correct xformers version for PyTorch 2.6.0
if [ "$FLASH_ATTN_INSTALLED" = true ]; then
    echo "⚡ Installing correct xformers version for PyTorch 2.6.0..."
    # xformers versions compatible with PyTorch 2.6.0
    XFORMERS_VERSIONS=("0.0.30" "0.0.29" "0.0.28" "0.0.27")
    XFORMERS_INSTALLED=false
    
    for version in "${XFORMERS_VERSIONS[@]}"; do
        echo "🔄 Trying xformers==$version..."
        if pip install xformers==$version; then
            echo "✅ xformers $version installed successfully"
            XFORMERS_INSTALLED=true
            break
        else
            echo "❌ xformers $version failed"
        fi
    done
    
    if [ "$XFORMERS_INSTALLED" = false ]; then
        echo "⚠️ All xformers versions failed, trying pre-built wheel..."
        # Try installing from pre-built wheel without strict dependency checking
        if pip install xformers --no-deps --force-reinstall; then
            echo "✅ xformers installed from pre-built wheel"
        else
            echo "❌ All xformers installation methods failed"
            echo "🔄 Continuing without xformers (will use standard attention)"
            export DISABLE_XFORMERS=1
        fi
    fi
else
    echo "⚠️ Installing xformers without flash-attn..."
    # Try xformers without flash-attn dependency
    XFORMERS_VERSIONS=("0.0.30" "0.0.29" "0.0.28" "0.0.27")
    XFORMERS_INSTALLED=false
    
    for version in "${XFORMERS_VERSIONS[@]}"; do
        echo "🔄 Trying xformers==$version (no flash-attn)..."
        if pip install xformers==$version; then
            echo "✅ xformers $version installed successfully"
            XFORMERS_INSTALLED=true
            break
        else
            echo "❌ xformers $version failed"
        fi
    done
    
    if [ "$XFORMERS_INSTALLED" = false ]; then
        echo "❌ All xformers versions failed"
        export DISABLE_XFORMERS=1
    fi
fi

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