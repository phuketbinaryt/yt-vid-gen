# Hugging Face Authentication Guide

## Overview

The FramePack API uses several models from Hugging Face, including some that are **gated** and require authentication and access approval. This guide explains how to set up authentication and access these models.

## 🔐 Authentication Setup

### Step 1: Get Your Hugging Face Token

1. **Create Account**: Sign up at [huggingface.co](https://huggingface.co) if you don't have an account
2. **Generate Token**: Go to [Settings > Access Tokens](https://huggingface.co/settings/tokens)
3. **Create New Token**: Click "New token" with **Write** permissions
4. **Copy Token**: Save your token securely (starts with `hf_...`)

### Step 2: Request Access to Gated Models

Some models require explicit access approval:

#### FLUX.1-Redux-dev (Gated Model)
- **Model**: `black-forest-labs/FLUX.1-Redux-dev`
- **Access**: Visit [model page](https://huggingface.co/black-forest-labs/FLUX.1-Redux-dev) and click "Request access"
- **Purpose**: Advanced image encoding for higher quality video generation
- **Fallback**: `google/siglip-so400m-patch14-384` (automatically used if access denied)

#### Other Models (Public)
- **HunyuanVideo**: `hunyuanvideo-community/HunyuanVideo` ✅ Public
- **FramePack**: `lllyasviel/FramePackI2V_HY` ✅ Public  
- **FramePack-F1**: `lllyasviel/FramePack_F1_I2V_HY_20250503` ✅ Public

## 🚀 Deployment with Authentication

### RunPod Deployment

Set the `HF_TOKEN` environment variable before running the deployment script:

```bash
# Set your Hugging Face token
export HF_TOKEN="hf_your_token_here"

# Run deployment script
bash <(curl -s https://raw.githubusercontent.com/phuketbinaryt/yt-vid-gen/main/runpod-startup.sh)
```

### Manual Environment Setup

Add your token to the `.env` file:

```bash
# Add to .env file
HF_TOKEN=hf_your_token_here
```

### Docker Deployment

Pass the token as an environment variable:

```bash
docker run -e HF_TOKEN="hf_your_token_here" your-framepack-image
```

## 🔄 Fallback Mechanism

The API includes automatic fallback for gated models:

### Image Encoder Fallback Chain

1. **Primary**: `lllyasviel/flux_redux_bfl` (if authenticated)
2. **Fallback 1**: `google/siglip-so400m-patch14-384` (public)
3. **Fallback 2**: `google/siglip-base-patch16-224` (public)

### Error Handling

If authentication fails, the API will:
- ✅ Continue running with fallback models
- ⚠️ Log warnings about unavailable models
- 📝 Provide clear error messages
- 🔄 Automatically retry with alternative models

## 🧪 Testing Authentication

### Check Authentication Status

```python
from huggingface_hub import whoami

try:
    user_info = whoami()
    print(f"✅ Authenticated as: {user_info['name']}")
except Exception as e:
    print(f"❌ Authentication failed: {e}")
```

### Test Model Access

```python
from transformers import SiglipImageProcessor

# Test gated model access
try:
    processor = SiglipImageProcessor.from_pretrained("black-forest-labs/FLUX.1-Redux-dev")
    print("✅ FLUX Redux model accessible")
except Exception as e:
    print(f"⚠️ FLUX Redux not accessible: {e}")
    
# Test fallback model
try:
    processor = SiglipImageProcessor.from_pretrained("google/siglip-so400m-patch14-384")
    print("✅ Fallback model accessible")
except Exception as e:
    print(f"❌ Fallback model failed: {e}")
```

## 🎯 Model Quality Comparison

### FLUX Redux (Gated) vs SigLIP (Public)

| Feature | FLUX Redux | SigLIP Alternative |
|---------|------------|-------------------|
| **Quality** | ⭐⭐⭐⭐⭐ Highest | ⭐⭐⭐⭐ High |
| **Speed** | ⭐⭐⭐⭐ Fast | ⭐⭐⭐⭐⭐ Fastest |
| **VRAM** | ~2GB | ~1.5GB |
| **Access** | Requires approval | Public |
| **Features** | Advanced encoding | Standard encoding |

### Performance Impact

- **With FLUX Redux**: Best possible video quality, optimal for production
- **With SigLIP**: Excellent quality, faster processing, no authentication needed
- **Difference**: ~5-10% quality improvement with FLUX Redux

## 🛠️ Troubleshooting

### Common Issues

#### 1. "401 Unauthorized" Error
```
GatedRepoError: 401 Client Error
```
**Solution**: 
- Check your HF_TOKEN is valid
- Request access to the gated model
- Wait for approval (can take 1-24 hours)

#### 2. "Token not found" Error
```
⚠️ HF_TOKEN not found
```
**Solution**:
- Set HF_TOKEN environment variable
- Add token to .env file
- Pass token during deployment

#### 3. Model Loading Fails
```
❌ Failed to load image encoder
```
**Solution**:
- Check internet connection
- Verify model names are correct
- Try fallback models manually

### Debug Commands

```bash
# Check environment variable
echo $HF_TOKEN

# Test authentication
python -c "from huggingface_hub import whoami; print(whoami())"

# Check model access
python -c "
from transformers import SiglipImageProcessor
try:
    SiglipImageProcessor.from_pretrained('black-forest-labs/FLUX.1-Redux-dev')
    print('✅ FLUX Redux accessible')
except:
    print('⚠️ FLUX Redux not accessible')
"
```

## 📚 Additional Resources

- [Hugging Face Documentation](https://huggingface.co/docs)
- [Model Access Requests](https://huggingface.co/docs/hub/models-gated)
- [Authentication Guide](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- [FramePack Repository](https://github.com/lllyasviel/FramePack)

## 💡 Best Practices

1. **Security**: Never commit tokens to version control
2. **Environment**: Use environment variables for tokens
3. **Fallbacks**: Always implement fallback mechanisms
4. **Testing**: Test both authenticated and non-authenticated scenarios
5. **Monitoring**: Log authentication status for debugging
6. **Updates**: Regularly check for new model versions

## 🎉 Success Indicators

When everything is working correctly, you should see:

```
✅ Successfully authenticated with Hugging Face
✅ Successfully loaded FLUX Redux image encoder
🎉 All imports successful!
✅ FramePack API is running!
```

If you see fallback messages, the API is still functional but using alternative models:

```
⚠️ FLUX Redux model is gated, trying alternative image encoder...
✅ Successfully loaded alternative SigLIP image encoder