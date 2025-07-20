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

# Install Python dependencies
pip install -r requirements.txt

# Install PyTorch with CUDA (if not already installed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

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

# Start the API
echo "🎬 Starting FramePack API on port 8000..."
python main.py &

# Keep container running
echo "✅ FramePack API is running!"
echo "📡 API URL: http://localhost:8000"
echo "📚 Documentation: http://localhost:8000/docs"
echo "❤️ Health Check: http://localhost:8000/health"

# Show logs
tail -f /dev/null