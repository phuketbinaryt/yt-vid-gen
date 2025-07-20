# Aspect Ratio Guide

The FramePack API now supports custom aspect ratios for video generation, giving you full control over video dimensions.

## 🎯 Overview

Instead of auto-detecting dimensions from the input image, you can now specify the exact aspect ratio for your videos using the `aspect_ratio` parameter.

## 📐 Supported Aspect Ratios

### **Standard Ratios**
| Aspect Ratio | Dimensions | Best For | Performance |
|--------------|------------|----------|-------------|
| `"1:1"` | 640x640 | Square videos, Instagram posts | ⚡ Fastest |
| `"16:9"` | 1024x576 | Landscape, YouTube, TV | 🚀 Fast |
| `"9:16"` | 576x1024 | Portrait, TikTok, Instagram Stories | 🚀 Fast |
| `"4:3"` | 768x576 | Traditional TV, presentations | 🔄 Medium |
| `"3:2"` | 768x512 | Photography standard | 🔄 Medium |
| `"2:3"` | 512x768 | Portrait photography | 🔄 Medium |
| `"3:4"` | 576x768 | Portrait traditional | 🔄 Medium |

### **Cinematic Ratios**
| Aspect Ratio | Dimensions | Best For | Performance |
|--------------|------------|----------|-------------|
| `"21:9"` | 1344x576 | Ultra-wide cinematic | 🐌 Slower |
| `"2.35:1"` | 1200x512 | Anamorphic widescreen | 🐌 Slower |

### **Custom Ratios**
You can also specify custom ratios like `"5:4"`, `"7:5"`, etc. The API will calculate optimal dimensions automatically.

## 🚀 Usage Examples

### **Portrait Video (TikTok/Instagram Stories)**
```json
{
  "prompt": "A person dancing in a colorful room",
  "aspect_ratio": "9:16",
  "duration": 5.0,
  "steps": 20
}
```

### **Landscape Video (YouTube)**
```json
{
  "prompt": "Beautiful sunset over mountains",
  "aspect_ratio": "16:9",
  "duration": 10.0,
  "use_teacache": true
}
```

### **Square Video (Instagram Post)**
```json
{
  "prompt": "Product showcase rotating",
  "aspect_ratio": "1:1",
  "duration": 3.0,
  "steps": 18
}
```

### **Cinematic Video**
```json
{
  "prompt": "Epic space battle scene",
  "aspect_ratio": "21:9",
  "duration": 15.0,
  "gpu_memory_preservation": 4.0
}
```

### **Custom Aspect Ratio**
```json
{
  "prompt": "Artistic composition",
  "aspect_ratio": "5:4",
  "duration": 8.0
}
```

## 📊 Performance Comparison

| Aspect Ratio | Pixel Count | Relative Speed | VRAM Usage | Recommended For |
|--------------|-------------|----------------|------------|-----------------|
| **1:1** (640x640) | 409K | 100% (baseline) | Lowest | Quick tests, social media |
| **16:9** (1024x576) | 590K | 85% | Medium | YouTube, general use |
| **9:16** (576x1024) | 590K | 85% | Medium | Mobile content |
| **4:3** (768x576) | 442K | 95% | Low-Medium | Traditional content |
| **21:9** (1344x576) | 774K | 70% | Higher | Cinematic content |

## 🎛️ API Integration

### **Request Format**
```bash
curl -X POST "http://your-api/api/v1/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Your video description",
    "aspect_ratio": "16:9",
    "duration": 5.0,
    "image_url": "https://example.com/image.jpg"
  }'
```

### **Response Format**
```json
{
  "job_id": "abc123",
  "status": "completed",
  "video_url": "http://your-api/static/abc123_final.mp4",
  "metadata": {
    "video_dimensions": "1024x576",
    "aspect_ratio_used": "16:9",
    "model_used": "FramePack",
    "parameters": {
      "aspect_ratio": "16:9",
      "prompt": "Your video description"
    }
  }
}
```

## ⚡ Speed Optimization Tips

### **For Maximum Speed**
- Use `"1:1"` (square) aspect ratio
- Set `steps: 18-20`
- Enable `use_teacache: true`
- Lower `gpu_memory_preservation` to 3.0-4.0

### **For Best Quality/Speed Balance**
- Use `"16:9"` or `"9:16"`
- Set `steps: 22-25`
- Keep `use_teacache: true`

### **For Cinematic Quality**
- Use `"21:9"` or `"2.35:1"`
- Set `steps: 25-30`
- Increase `gpu_memory_preservation` to 6.0+

## 🔧 Technical Details

### **Dimension Calculation**
- All dimensions are automatically aligned to 8-pixel boundaries (required by FramePack)
- Maximum dimension is capped at 1920 pixels
- Minimum dimension is 256 pixels
- Base resolution is 640 pixels for custom ratios

### **Memory Usage**
- Square videos (1:1) use the least memory
- Ultra-wide videos (21:9) use the most memory
- Portrait and landscape videos (9:16, 16:9) are well-optimized

### **Backward Compatibility**
- If no `aspect_ratio` is specified, the API falls back to auto-detection from input image
- Existing requests without `aspect_ratio` continue to work unchanged

## 🎨 Creative Use Cases

### **Social Media Content**
- **Instagram Posts**: `"1:1"` - Perfect square format
- **Instagram Stories**: `"9:16"` - Full-screen mobile
- **TikTok**: `"9:16"` - Optimized for mobile viewing
- **YouTube**: `"16:9"` - Standard widescreen

### **Professional Content**
- **Presentations**: `"4:3"` - Traditional format
- **Cinematic**: `"21:9"` - Ultra-wide for films
- **Photography**: `"3:2"` - Standard photo ratio

### **Custom Projects**
- **Art Installations**: Custom ratios like `"5:4"`, `"7:5"`
- **Digital Signage**: Custom ratios for specific displays
- **Mobile Apps**: Custom ratios for specific UI requirements

## 🚨 Important Notes

1. **Input Image Handling**: The input image will be automatically resized and center-cropped to match the specified aspect ratio
2. **Performance Impact**: Larger dimensions (more pixels) will take longer to generate
3. **Memory Requirements**: Ultra-wide ratios may require more GPU memory
4. **Quality**: All aspect ratios maintain the same quality standards - only dimensions change

## 🔍 Troubleshooting

### **Invalid Aspect Ratio Error**
```json
{
  "error": "Invalid aspect ratio '16:10'. Supported ratios: 1:1, 16:9, 9:16, 4:3, 3:2, 21:9, 2.35:1 or custom format like '16:9'"
}
```
**Solution**: Use one of the supported ratios or ensure custom ratio format is correct (e.g., "16:10")

### **Memory Issues with Large Ratios**
If you get CUDA out of memory errors with ultra-wide ratios:
- Increase `gpu_memory_preservation` to 6.0 or higher
- Reduce `steps` to 18-20
- Consider using a smaller custom ratio

### **Slow Generation**
For faster generation with custom aspect ratios:
- Use predefined ratios when possible (they're optimized)
- Choose ratios closer to 1:1 for better performance
- Enable `use_teacache: true`