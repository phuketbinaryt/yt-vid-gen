#!/bin/bash

# Clean Ubuntu FramePack API deployment script
echo "🚀 Starting FramePack API deployment on clean Ubuntu server..."

# Update system and install basic dependencies
echo "📦 Installing system dependencies..."
apt update && apt upgrade -y
apt install -y git curl wget build-essential python3-dev python3-pip python3-venv python3-full redis-server

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
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
else
    echo "⚠️ No requirements.txt found in FramePack, installing known dependencies..."
    # Install known FramePack dependencies
    pip install diffusers transformers accelerate
    pip install pillow opencv-python av
    pip install numpy scipy einops
    pip install safetensors sentencepiece
    pip install torchsde
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
FLUX_REDUX_MODEL_PATH=black-forest-labs/FLUX.1-Redux-dev
FRAMEPACK_MODEL_PATH=lllyasviel/FramePack
FRAMEPACK_F1_MODEL_PATH=lllyasviel/FramePack-F1

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

# Set Python path for FramePack modules
echo "🐍 Setting up Python path..."
export PYTHONPATH="/workspace/yt-vid-gen/FramePack:$PYTHONPATH"

# Start Redis server
echo "🔴 Starting Redis server..."
redis-server --daemonize yes --bind 0.0.0.0 --port 6379

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