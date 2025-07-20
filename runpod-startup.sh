#!/bin/bash

# Clean Ubuntu FramePack API deployment script
echo "🚀 Starting FramePack API deployment on clean Ubuntu server..."

# Update system and install basic dependencies
echo "📦 Installing system dependencies..."
apt update && apt upgrade -y
apt install -y git curl wget build-essential python3-dev python3-pip python3-venv redis-server

# Install CUDA toolkit if not present (for GPU support)
echo "🔧 Setting up CUDA environment..."
if ! command -v nvcc &> /dev/null; then
    echo "Installing CUDA toolkit..."
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt update
    apt install -y cuda-toolkit-12-4
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
fi

# Set up Python virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv /workspace/framepack-env
source /workspace/framepack-env/bin/activate
pip install --upgrade pip setuptools wheel

# Clone our API repository
echo "📥 Cloning FramePack API repository..."
if [ ! -d "/workspace/yt-vid-gen" ]; then
    cd /workspace
    git clone https://github.com/phuketbinaryt/yt-vid-gen.git
    cd yt-vid-gen
else
    cd /workspace/yt-vid-gen
    git pull origin main
fi

# Clone original FramePack repository
echo "📥 Cloning original FramePack repository..."
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

# Install FramePack dependencies following official requirements
echo "📦 Installing FramePack dependencies from official repository..."
cd FramePack

# Install PyTorch with CUDA support (following FramePack requirements)
echo "🔥 Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# Install FramePack requirements
echo "📋 Installing FramePack requirements..."
# Always install core dependencies first
pip install diffusers transformers accelerate==1.0.1
pip install pillow opencv-python av
pip install numpy scipy einops
pip install safetensors sentencepiece
pip install torchsde

# Then install from requirements.txt if it exists, but fix problematic versions
if [ -f "requirements.txt" ]; then
    echo "📋 Installing additional requirements from FramePack requirements.txt..."
    # Fix accelerate version issue - replace 1.6.0 with 1.0.1 if it exists
    if grep -q "accelerate==1.6.0" requirements.txt; then
        echo "🔧 Fixing accelerate version from 1.6.0 to 1.0.1..."
        sed -i 's/accelerate==1.6.0/accelerate==1.0.1/g' requirements.txt
    fi
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found in FramePack"
fi

# Install attention optimizations
echo "⚡ Installing attention optimizations..."
pip install xformers
pip install flash-attn --no-build-isolation

cd ..

# Install our API dependencies
echo "🌐 Installing API dependencies..."
pip install fastapi uvicorn redis celery
pip install pydantic pydantic-settings
pip install python-multipart aiofiles
pip install starlette httpx requests
pip install python-dotenv

# Install Hugging Face Hub for authentication
echo "🤗 Installing Hugging Face Hub..."
pip install huggingface_hub

# Create environment configuration
echo "⚙️ Setting up environment configuration..."
cat > .env << EOF
# FramePack API Configuration
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379/0

# GPU Configuration
GPU_DEVICE=cuda:0
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2
GPU_MEMORY_PRESERVATION=6.0

# Model Paths (using HuggingFace cache)
HUNYUAN_MODEL_PATH=hunyuanvideo-community/HunyuanVideo
FLUX_REDUX_MODEL_PATH=lllyasviel/flux_redux_bfl
FRAMEPACK_MODEL_PATH=lllyasviel/FramePackI2V_HY
FRAMEPACK_F1_MODEL_PATH=lllyasviel/FramePack_F1_I2V_HY_20250503

# Hugging Face Authentication (set your token here for gated models)
# HF_TOKEN=your_huggingface_token_here

# File Storage
UPLOAD_DIR=/workspace/uploads
OUTPUT_DIR=/workspace/outputs
TEMP_DIR=/workspace/temp

# Performance Settings
DEFAULT_DURATION=5
DEFAULT_STEPS=30
DEFAULT_CFG_SCALE=7.0
DEFAULT_DISTILLED_CFG_SCALE=3.5
DEFAULT_CFG_RESCALE=0.7
DEFAULT_SEED=42
DEFAULT_MP4_CRF=18
DEFAULT_LATENT_WINDOW_SIZE=16

# Optional API Key (uncomment to enable)
# API_KEY=your-secret-key-here
EOF

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p /workspace/uploads /workspace/outputs /workspace/temp

# Set up HuggingFace cache
echo "🤗 Setting up HuggingFace cache..."
export HF_HOME=/workspace/hf_cache
mkdir -p $HF_HOME

# Setup git credential helper to avoid warnings
echo "🔧 Setting up git credential helper..."
git config --global credential.helper store

# Setup Hugging Face authentication
echo "🔐 Setting up Hugging Face authentication..."
if [ -n "$HF_TOKEN" ]; then
    echo "Found HF_TOKEN environment variable, authenticating..."
    python -c "
import os
from huggingface_hub import login
try:
    token = os.environ.get('HF_TOKEN')
    if token:
        login(token=token, add_to_git_credential=True)
        print('✅ Successfully authenticated with Hugging Face')
    else:
        print('⚠️ HF_TOKEN is empty')
except Exception as e:
    print(f'⚠️ Failed to authenticate with Hugging Face: {e}')
    print('Continuing without authentication - some models may not be available')
"
else
    echo "⚠️ HF_TOKEN not found. Some gated models may not be accessible."
    echo "💡 To use gated models like FLUX Redux:"
    echo "   1. Get your token from: https://huggingface.co/settings/tokens"
    echo "   2. Set HF_TOKEN environment variable before running this script"
    echo "   3. Request access to gated models at their respective pages"
    echo ""
    echo "🔄 Continuing with fallback models for image encoding..."
fi

# Set Python path for FramePack modules
echo "🐍 Setting up Python path..."
export PYTHONPATH="/workspace/yt-vid-gen/FramePack:$PYTHONPATH"

# Start Redis server
echo "🔴 Starting Redis server..."
if command -v redis-server &> /dev/null; then
    redis-server --daemonize yes --bind 0.0.0.0 --port 6379
else
    echo "⚠️ Redis server not found, installing..."
    apt install -y redis-server
    redis-server --daemonize yes --bind 0.0.0.0 --port 6379
fi

# Wait for Redis to start
sleep 3

# Test FramePack installation
echo "🧪 Testing FramePack installation..."
cd /workspace/yt-vid-gen
python -c "
import sys
sys.path.insert(0, './FramePack')
try:
    import torch
    print(f'✅ PyTorch {torch.__version__} with CUDA: {torch.cuda.is_available()}')
    
    from diffusers import AutoencoderKLHunyuanVideo
    print('✅ Diffusers import successful')
    
    from transformers import LlamaModel, CLIPTextModel
    print('✅ Transformers import successful')
    
    # Test FramePack modules
    from diffusers_helper.hunyuan import encode_prompt_conds
    print('✅ FramePack diffusers_helper import successful')
    
    print('🎉 All imports successful!')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo "✅ FramePack installation test passed!"
else
    echo "❌ FramePack installation test failed!"
    exit 1
fi

# Create activation script for services
echo "📝 Creating virtual environment activation script..."
cat > /workspace/activate_env.sh << 'EOF'
#!/bin/bash
source /workspace/framepack-env/bin/activate
export PYTHONPATH="/workspace/yt-vid-gen/FramePack:$PYTHONPATH"
export HF_HOME=/workspace/hf_cache
cd /workspace/yt-vid-gen
exec "$@"
EOF
chmod +x /workspace/activate_env.sh

# Start the API server
echo "🎬 Starting FramePack API server..."
cd /workspace/yt-vid-gen

# Start Celery worker in background
echo "👷 Starting Celery worker..."
/workspace/activate_env.sh celery -A framepack_worker worker --loglevel=info --concurrency=1 &

# Start FastAPI server
echo "🌐 Starting FastAPI server..."
/workspace/activate_env.sh python main.py &

# Keep container running and show logs
echo "✅ FramePack API is running!"
echo "📡 API URL: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"
echo ""
echo "🔍 Monitoring logs..."

# Show real-time logs
tail -f /dev/null