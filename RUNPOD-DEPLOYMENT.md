# RunPod Deployment Guide for FramePack API

## 🎯 **Recommended RunPod Templates**

Based on your screenshot, here are the best templates for FramePack API deployment:

### **1. 🥇 BEST CHOICE: "Runpod Pytorch 2.4.0" (Official)**
- ✅ **PyTorch 2.4.0** pre-installed
- ✅ **CUDA 12.1+** support
- ✅ **Python 3.10+** environment
- ✅ **Official RunPod template** (well-maintained)
- ✅ **Perfect for AI/ML workloads**

### **2. 🥈 ALTERNATIVE: "Runpod Pytorch 2.1" (Official)**
- ✅ **PyTorch 2.1** (compatible)
- ✅ **CUDA support**
- ✅ **Stable and tested**

### **3. 🥉 BACKUP: "Ubuntu 20.04 server" (Community)**
- ✅ **Clean Ubuntu base**
- ⚠️ **Requires manual PyTorch installation**
- ✅ **Full control over environment**

## 🖥️ **GPU Requirements**

### **Minimum (Budget):**
- **RTX 3070/3080** (8-12GB VRAM)
- **Generation Speed**: 4-8 seconds/frame
- **Concurrent Jobs**: 1

### **Recommended (Performance):**
- **RTX 4090** (24GB VRAM)
- **Generation Speed**: 1.5-2.5 seconds/frame
- **Concurrent Jobs**: 2

### **Enterprise (Maximum):**
- **H100/A100** (80GB VRAM)
- **Generation Speed**: 0.5-1 second/frame
- **Concurrent Jobs**: 4+

## 🚀 **Quick Deployment Steps**

### **Step 1: Create RunPod Instance**
1. Select **"Runpod Pytorch 2.4.0"** template
2. Choose GPU (RTX 4090 recommended)
3. Set volume size: **50GB+**
4. Expose ports: **8000** (API), **6379** (Redis)

### **Step 2: Connect and Deploy**
```bash
# Connect to your RunPod instance via SSH or web terminal

# Download and run the startup script
wget https://raw.githubusercontent.com/phuketbinaryt/yt-vid-gen/main/runpod-startup.sh
chmod +x runpod-startup.sh
./runpod-startup.sh
```

### **Step 3: Access Your API**
- **API URL**: `https://your-pod-id-8000.proxy.runpod.net`
- **Documentation**: `https://your-pod-id-8000.proxy.runpod.net/docs`
- **Health Check**: `https://your-pod-id-8000.proxy.runpod.net/health`

## 🔧 **Manual Deployment (Alternative)**

If you prefer manual setup:

```bash
# 1. Clone repository
cd /workspace
git clone https://github.com/phuketbinaryt/yt-vid-gen.git
cd yt-vid-gen

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install Redis
apt update && apt install -y redis-server

# 4. Start Redis
redis-server --daemonize yes

# 5. Configure environment
cp .env.example .env

# 6. Start API
python main.py
```

## 🌐 **RunPod Network Configuration**

### **Port Mapping:**
- **8000**: FramePack API
- **6379**: Redis (internal)

### **Environment Variables:**
```env
API_HOST=0.0.0.0
API_PORT=8000
REDIS_URL=redis://localhost:6379/0
GPU_DEVICE=cuda:0
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2
```

## 📊 **Performance Optimization for RunPod**

### **For RTX 4090:**
```env
HIGH_VRAM_THRESHOLD=20.0
MAX_CONCURRENT_JOBS=2
GPU_MEMORY_PRESERVATION=6.0
```

### **For H100/A100:**
```env
HIGH_VRAM_THRESHOLD=60.0
MAX_CONCURRENT_JOBS=4
GPU_MEMORY_PRESERVATION=8.0
```

### **For RTX 3070/3080:**
```env
HIGH_VRAM_THRESHOLD=10.0
MAX_CONCURRENT_JOBS=1
GPU_MEMORY_PRESERVATION=4.0
```

## 🧪 **Testing Your Deployment**

### **1. Health Check:**
```bash
curl https://your-pod-id-8000.proxy.runpod.net/health
```

### **2. Generate Video:**
```bash
curl -X POST "https://your-pod-id-8000.proxy.runpod.net/api/v1/generate/text-to-video" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A beautiful sunset over mountains",
    "duration": 5.0,
    "use_f1_model": false
  }'
```

### **3. Check Job Status:**
```bash
curl "https://your-pod-id-8000.proxy.runpod.net/api/v1/jobs/{job_id}/status"
```

## 💰 **Cost Optimization**

### **Spot Instances:**
- Use **Spot pricing** for 50-70% savings
- Good for development and testing
- May be interrupted during high demand

### **On-Demand:**
- Use for production workloads
- Guaranteed availability
- Higher cost but reliable

### **Storage:**
- **Network Volume**: Persistent, slower
- **Container Disk**: Faster, temporary
- **Recommendation**: Use network volume for models, container disk for temp files

## 🔒 **Security Configuration**

### **Enable API Key (Recommended for Production):**
```env
API_KEY=your-secure-api-key-here
```

### **Restrict Access:**
```env
ALLOWED_ORIGINS=["https://yourdomain.com"]
```

## 📝 **Monitoring and Logs**

### **View API Logs:**
```bash
# If using startup script
tail -f /workspace/yt-vid-gen/api.log

# If running manually
# Logs will appear in terminal
```

### **Monitor GPU Usage:**
```bash
nvidia-smi -l 1
```

### **Check Redis:**
```bash
redis-cli ping
```

## 🚨 **Troubleshooting**

### **Common Issues:**

1. **Out of Memory:**
   - Reduce `MAX_CONCURRENT_JOBS`
   - Increase `GPU_MEMORY_PRESERVATION`

2. **Model Download Fails:**
   - Check internet connection
   - Ensure sufficient disk space (30GB+)

3. **Redis Connection Error:**
   - Restart Redis: `redis-server --daemonize yes`
   - Check Redis status: `redis-cli ping`

4. **API Not Accessible:**
   - Verify port 8000 is exposed
   - Check firewall settings
   - Ensure API_HOST=0.0.0.0

## 📞 **Support**

- **GitHub Issues**: https://github.com/phuketbinaryt/yt-vid-gen/issues
- **API Documentation**: `/docs` endpoint
- **Health Check**: `/health` endpoint

## 🎉 **Success Indicators**

Your deployment is successful when:
- ✅ Health check returns `{"status": "healthy"}`
- ✅ GPU is detected and available
- ✅ Models load without errors
- ✅ Test video generation completes
- ✅ API documentation is accessible

**Happy video generating! 🎬**