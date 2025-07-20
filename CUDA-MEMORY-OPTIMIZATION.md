# CUDA Memory Optimization Guide

## 🚨 CUDA Out of Memory Fix

This document explains the CUDA memory optimizations implemented to resolve the "CUDA out of memory" error encountered during video generation on RTX 4090 (24GB VRAM).

## 🔧 Implemented Solutions

### 1. PyTorch Memory Configuration
- **Environment Variable**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
- **Purpose**: Reduces memory fragmentation by allowing PyTorch to expand memory segments dynamically
- **Implementation**: Set in both startup script and worker code

### 2. Aggressive Memory Management
- **Frequent Cache Clearing**: `torch.cuda.empty_cache()` called at critical points
- **Immediate Tensor Cleanup**: `del tensor_name` followed by cache clearing
- **Model Offloading**: Enhanced memory preservation with 10GB reserved

### 3. Chunked VAE Decoding
- **Large Sequences**: Process in 16-frame chunks for sequences > 32 frames
- **Emergency Fallback**: 4-frame chunks if 16-frame chunks fail
- **Last Resort**: Frame-by-frame processing for extreme memory pressure

### 4. Enhanced Error Recovery
- **CUDA OOM Handling**: Automatic fallback to smaller chunk sizes
- **Memory Monitoring**: Proactive cleanup before memory-intensive operations
- **Graceful Degradation**: Multiple fallback strategies

## 📊 Memory Usage Patterns

### Before Optimization
```
GPU Memory: 21.13 GiB allocated by PyTorch
Free Memory: 2.13 GiB
Allocation Request: 2.97 GiB → FAILURE
```

### After Optimization
```
GPU Memory: Dynamically managed with expandable segments
Chunked Processing: Reduces peak memory usage
Aggressive Cleanup: Maximizes available memory
```

## 🛠️ Key Implementation Points

### 1. Worker Initialization
```python
# Set memory configuration before importing torch
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
```

### 2. VAE Decoding with Chunking
```python
if real_history_latents.shape[2] > 32:  # Large sequence
    chunk_size = 16
    for i in range(0, frames, chunk_size):
        torch.cuda.empty_cache()  # Clear before each chunk
        chunk_pixels = vae_decode(chunk_latents, self.vae).cpu()
        # Process and cleanup immediately
```

### 3. Emergency Recovery
```python
except torch.cuda.OutOfMemoryError:
    torch.cuda.empty_cache()
    # Try with smaller chunks (4 frames)
    # Fallback to frame-by-frame if needed
```

## 🎯 Performance Impact

### Memory Efficiency
- **Reduced Peak Usage**: Chunked processing prevents memory spikes
- **Better Fragmentation**: Expandable segments reduce wasted memory
- **Faster Recovery**: Aggressive cleanup enables rapid memory reuse

### Generation Quality
- **No Quality Loss**: Chunked processing maintains full quality
- **Consistent Results**: Same output as non-chunked processing
- **Reliable Operation**: Handles various video lengths and resolutions

## 🔍 Monitoring and Debugging

### Memory Tracking
```python
free_mem_gb = get_cuda_free_memory_gb(gpu)
print(f'Free VRAM: {free_mem_gb} GB')
```

### Error Logging
```python
except torch.cuda.OutOfMemoryError as e:
    print(f"⚠️ CUDA OOM during VAE decode, attempting recovery: {e}")
```

## 📋 Deployment Checklist

- ✅ PyTorch memory configuration set in startup script
- ✅ Worker code includes memory optimizations
- ✅ Chunked VAE decoding implemented
- ✅ Emergency fallback strategies in place
- ✅ Comprehensive error handling added
- ✅ Memory cleanup at all critical points

## 🚀 Expected Results

With these optimizations, the FramePack API should successfully generate videos on RTX 4090 without CUDA out of memory errors, even for longer sequences and higher resolutions.

## 🔧 Troubleshooting

If CUDA OOM errors persist:

1. **Reduce Chunk Size**: Modify `chunk_size = 8` for even smaller chunks
2. **Increase Memory Preservation**: Set `preserved_memory_gb=12` or higher
3. **Enable VAE Tiling**: Ensure `self.vae.enable_tiling()` is active
4. **Monitor GPU Usage**: Use `nvidia-smi` to track memory usage patterns

## 📚 References

- [PyTorch CUDA Memory Management](https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
- [CUDA Memory Optimization Best Practices](https://pytorch.org/docs/stable/notes/cuda.html#memory-management)
- [Diffusers Memory Optimization](https://huggingface.co/docs/diffusers/optimization/memory)