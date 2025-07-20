import os
import sys

# Set PyTorch CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import traceback
import einops
import numpy as np
import base64
import requests
from io import BytesIO
from PIL import Image
from datetime import datetime
import time

# Disable flash-attn if environment variable is set
if os.environ.get('DISABLE_FLASH_ATTN'):
    print("⚠️ Flash attention disabled via environment variable")
    # Monkey patch to disable flash attention in transformers
    import transformers.utils.import_utils
    original_is_flash_attn_available = transformers.utils.import_utils.is_flash_attn_2_available
    transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
    transformers.utils.import_utils.is_flash_attn_available = lambda: False

# Disable xformers if environment variable is set
if os.environ.get('DISABLE_XFORMERS'):
    print("⚠️ xformers disabled via environment variable")
    # Monkey patch to disable xformers in diffusers
    try:
        import diffusers.utils.import_utils
        diffusers.utils.import_utils.is_xformers_available = lambda: False
    except ImportError:
        pass

# Add FramePack to path
import os
framepack_path = os.path.abspath('./FramePack')
if framepack_path not in sys.path:
    sys.path.insert(0, framepack_path)
    
# Also add the current directory to ensure relative imports work
current_dir = os.path.abspath('.')
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Try importing with flash-attn and xformers fallback
try:
    from diffusers import AutoencoderKLHunyuanVideo
    from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
    print("✅ Successfully imported diffusers and transformers")
except ImportError as e:
    error_str = str(e).lower()
    if "flash_attn" in error_str or "flash-attention" in error_str:
        print("⚠️ Flash attention import failed, disabling and retrying...")
        # Disable flash attention and retry
        os.environ['DISABLE_FLASH_ATTN'] = '1'
        import transformers.utils.import_utils
        transformers.utils.import_utils.is_flash_attn_2_available = lambda: False
        transformers.utils.import_utils.is_flash_attn_available = lambda: False
        
        try:
            from diffusers import AutoencoderKLHunyuanVideo
            from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
            print("✅ Successfully imported with flash attention disabled")
        except ImportError as e2:
            if "xformers" in str(e2).lower():
                print("⚠️ xformers import also failed, disabling both and retrying...")
                os.environ['DISABLE_XFORMERS'] = '1'
                import diffusers.utils.import_utils
                diffusers.utils.import_utils.is_xformers_available = lambda: False
                
                from diffusers import AutoencoderKLHunyuanVideo
                from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
                print("✅ Successfully imported with both flash attention and xformers disabled")
            else:
                raise
    elif "xformers" in error_str:
        print("⚠️ xformers import failed, disabling and retrying...")
        # Disable xformers and retry
        os.environ['DISABLE_XFORMERS'] = '1'
        import diffusers.utils.import_utils
        diffusers.utils.import_utils.is_xformers_available = lambda: False
        
        from diffusers import AutoencoderKLHunyuanVideo
        from transformers import LlamaModel, CLIPTextModel, LlamaTokenizerFast, CLIPTokenizer, SiglipImageProcessor, SiglipVisionModel
        print("✅ Successfully imported with xformers disabled")
    else:
        raise

# Import FramePack modules
from diffusers_helper.hunyuan import encode_prompt_conds, vae_decode, vae_encode, vae_decode_fake
from diffusers_helper.utils import save_bcthw_as_mp4, crop_or_pad_yield_mask, soft_append_bcthw, resize_and_center_crop, generate_timestamp
from diffusers_helper.models.hunyuan_video_packed import HunyuanVideoTransformer3DModelPacked
from diffusers_helper.pipelines.k_diffusion_hunyuan import sample_hunyuan
from diffusers_helper.memory import cpu, gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation, offload_model_from_device_for_memory_preservation, fake_diffusers_current_device, DynamicSwapInstaller, unload_complete_models, load_model_as_complete
from diffusers_helper.clip_vision import hf_clip_vision_encode
from diffusers_helper.bucket_tools import find_nearest_bucket

from config import settings
from job_manager import job_manager
from models import JobStatus, GenerationMode

# Import Celery
from celery import Celery

# Create Celery app
celery = Celery('framepack_worker')
celery.config_from_object('celery_config')

class FramePackWorker:
    def __init__(self):
        self.models_loaded = False
        self.high_vram = False
        self.setup_models()
    
    def setup_models(self):
        """Initialize and load all FramePack models"""
        try:
            print("Setting up FramePack models...")
            
            # Set HuggingFace cache directory
            os.environ['HF_HOME'] = os.path.abspath('./hf_download')
            
            # Check GPU memory
            free_mem_gb = get_cuda_free_memory_gb(gpu)
            self.high_vram = free_mem_gb > settings.HIGH_VRAM_THRESHOLD
            
            print(f'Free VRAM: {free_mem_gb} GB')
            print(f'High-VRAM Mode: {self.high_vram}')
            
            # Clear any existing GPU memory
            torch.cuda.empty_cache()
            
            # Load text encoders and tokenizers
            self.text_encoder = LlamaModel.from_pretrained(
                settings.HUNYUAN_MODEL_PATH, 
                subfolder='text_encoder', 
                torch_dtype=torch.float16
            ).cpu()
            
            self.text_encoder_2 = CLIPTextModel.from_pretrained(
                settings.HUNYUAN_MODEL_PATH, 
                subfolder='text_encoder_2', 
                torch_dtype=torch.float16
            ).cpu()
            
            self.tokenizer = LlamaTokenizerFast.from_pretrained(
                settings.HUNYUAN_MODEL_PATH, 
                subfolder='tokenizer'
            )
            
            self.tokenizer_2 = CLIPTokenizer.from_pretrained(
                settings.HUNYUAN_MODEL_PATH, 
                subfolder='tokenizer_2'
            )
            
            # Load VAE
            self.vae = AutoencoderKLHunyuanVideo.from_pretrained(
                settings.HUNYUAN_MODEL_PATH, 
                subfolder='vae', 
                torch_dtype=torch.float16
            ).cpu()
            
            # Load image encoder with fallback for gated repositories
            try:
                print(f"Loading image encoder from: {settings.FLUX_REDUX_MODEL_PATH}")
                self.feature_extractor = SiglipImageProcessor.from_pretrained(
                    settings.FLUX_REDUX_MODEL_PATH,
                    subfolder='feature_extractor'
                )
                
                self.image_encoder = SiglipVisionModel.from_pretrained(
                    settings.FLUX_REDUX_MODEL_PATH,
                    subfolder='image_encoder',
                    torch_dtype=torch.float16
                ).cpu()
                print("✅ Successfully loaded FLUX Redux image encoder")
                
            except Exception as e:
                error_str = str(e).lower()
                if "gated" in error_str or "401" in error_str or "unauthorized" in error_str:
                    print("⚠️ FLUX Redux model is gated, trying alternative image encoder...")
                    
                    # Try alternative SigLIP model that's publicly available
                    try:
                        alternative_model = "google/siglip-so400m-patch14-384"
                        print(f"Loading alternative image encoder: {alternative_model}")
                        
                        self.feature_extractor = SiglipImageProcessor.from_pretrained(alternative_model)
                        self.image_encoder = SiglipVisionModel.from_pretrained(
                            alternative_model,
                            torch_dtype=torch.float16
                        ).cpu()
                        print("✅ Successfully loaded alternative SigLIP image encoder")
                        
                    except Exception as e2:
                        print(f"❌ Failed to load alternative image encoder: {e2}")
                        # Try the most basic SigLIP model
                        try:
                            basic_model = "google/siglip-base-patch16-224"
                            print(f"Loading basic image encoder: {basic_model}")
                            
                            self.feature_extractor = SiglipImageProcessor.from_pretrained(basic_model)
                            self.image_encoder = SiglipVisionModel.from_pretrained(
                                basic_model,
                                torch_dtype=torch.float16
                            ).cpu()
                            print("✅ Successfully loaded basic SigLIP image encoder")
                            
                        except Exception as e3:
                            print(f"❌ All image encoder fallbacks failed: {e3}")
                            raise RuntimeError(
                                "Failed to load any image encoder. Please ensure you have access to the FLUX Redux model "
                                "or that alternative models are available. You may need to authenticate with Hugging Face "
                                "using 'huggingface-cli login' if using gated models."
                            )
                else:
                    print(f"❌ Unexpected error loading image encoder: {e}")
                    raise
            
            # Load transformer models (both standard and F1)
            self.transformer = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                settings.FRAMEPACK_MODEL_PATH, 
                torch_dtype=torch.bfloat16
            ).cpu()
            
            self.transformer_f1 = HunyuanVideoTransformer3DModelPacked.from_pretrained(
                settings.FRAMEPACK_F1_MODEL_PATH, 
                torch_dtype=torch.bfloat16
            ).cpu()
            
            # Set models to eval mode
            self.vae.eval()
            self.text_encoder.eval()
            self.text_encoder_2.eval()
            self.image_encoder.eval()
            self.transformer.eval()
            self.transformer_f1.eval()
            
            # Configure VAE for low VRAM
            if not self.high_vram:
                self.vae.enable_slicing()
                self.vae.enable_tiling()
            
            # Configure transformers
            self.transformer.high_quality_fp32_output_for_inference = True
            self.transformer_f1.high_quality_fp32_output_for_inference = True
            
            # Set dtypes
            self.transformer.to(dtype=torch.bfloat16)
            self.transformer_f1.to(dtype=torch.bfloat16)
            self.vae.to(dtype=torch.float16)
            self.image_encoder.to(dtype=torch.float16)
            self.text_encoder.to(dtype=torch.float16)
            self.text_encoder_2.to(dtype=torch.float16)
            
            # Disable gradients
            self.vae.requires_grad_(False)
            self.text_encoder.requires_grad_(False)
            self.text_encoder_2.requires_grad_(False)
            self.image_encoder.requires_grad_(False)
            self.transformer.requires_grad_(False)
            self.transformer_f1.requires_grad_(False)
            
            # Setup memory management - always use low VRAM mode for better stability
            print("Setting up dynamic memory management...")
            DynamicSwapInstaller.install_model(self.transformer, device=gpu)
            DynamicSwapInstaller.install_model(self.transformer_f1, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder, device=gpu)
            DynamicSwapInstaller.install_model(self.text_encoder_2, device=gpu)
            DynamicSwapInstaller.install_model(self.image_encoder, device=gpu)
            DynamicSwapInstaller.install_model(self.vae, device=gpu)
            
            # Don't load any models to GPU initially - they'll be loaded on demand
            print("Models configured for on-demand loading")
            
            self.models_loaded = True
            print("FramePack models loaded successfully!")
            
        except Exception as e:
            print(f"Error loading models: {e}")
            traceback.print_exc()
            raise
    
    def decode_image(self, image_b64: str) -> np.ndarray:
        """Decode base64 image to numpy array"""
        image_data = base64.b64decode(image_b64)
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        return np.array(image)
    
    def download_image_from_url(self, image_url: str) -> np.ndarray:
        """Download image from URL and convert to numpy array"""
        try:
            response = requests.get(image_url, timeout=30)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return np.array(image)
        except Exception as e:
            raise ValueError(f"Failed to download image from URL: {str(e)}")
    
    def generate_default_image(self, width: int = 640, height: int = 640) -> np.ndarray:
        """Generate a default image for text-to-video mode"""
        # Create a simple gradient or solid color image
        image = np.ones((height, width, 3), dtype=np.uint8) * 128  # Gray background
        return image
    
    @torch.no_grad()
    def process_job(self, job_id: str):
        """Process a video generation job"""
        try:
            # Get job data
            job_data = job_manager.get_job(job_id)
            if not job_data:
                raise ValueError(f"Job {job_id} not found")
            
            request_data = job_data['request_data']
            
            # Start job
            job_manager.start_job(job_id)
            
            # Extract parameters
            prompt = request_data['prompt']
            mode = request_data.get('mode', GenerationMode.IMAGE_TO_VIDEO)
            duration = request_data.get('duration', settings.DEFAULT_DURATION)
            seed = request_data.get('seed', settings.DEFAULT_SEED)
            steps = request_data.get('steps', settings.DEFAULT_STEPS)
            cfg = request_data.get('cfg_scale', settings.DEFAULT_CFG_SCALE)
            gs = request_data.get('distilled_cfg_scale', settings.DEFAULT_DISTILLED_CFG_SCALE)
            rs = request_data.get('cfg_rescale', settings.DEFAULT_CFG_RESCALE)
            use_teacache = request_data.get('use_teacache', True)
            gpu_memory_preservation = request_data.get('gpu_memory_preservation', settings.GPU_MEMORY_PRESERVATION)
            mp4_crf = request_data.get('mp4_crf', settings.DEFAULT_MP4_CRF)
            use_f1_model = request_data.get('use_f1_model', False)
            latent_window_size = settings.DEFAULT_LATENT_WINDOW_SIZE
            
            # Select transformer model
            transformer = self.transformer_f1 if use_f1_model else self.transformer
            
            # Calculate sections
            total_latent_sections = (duration * 30) / (latent_window_size * 4)
            total_latent_sections = int(max(round(total_latent_sections), 1))
            
            job_manager.update_progress(job_id, 5.0, "Processing input...")
            
            # Process input image
            if mode == GenerationMode.IMAGE_TO_VIDEO:
                if request_data.get('image'):
                    input_image = self.decode_image(request_data['image'])
                elif request_data.get('image_url'):
                    input_image = self.download_image_from_url(request_data['image_url'])
                else:
                    raise ValueError("Either 'image' or 'image_url' is required for image-to-video generation")
            else:
                # Generate default image for text-to-video
                input_image = self.generate_default_image()
            
            # Clean GPU memory aggressively
            torch.cuda.empty_cache()
            if not self.high_vram:
                unload_complete_models(
                    self.text_encoder, self.text_encoder_2, self.image_encoder, self.vae, transformer
                )
            torch.cuda.empty_cache()
            
            # Text encoding
            job_manager.update_progress(job_id, 10.0, "Encoding text prompt...")
            
            if not self.high_vram:
                fake_diffusers_current_device(self.text_encoder, gpu)
                load_model_as_complete(self.text_encoder_2, target_device=gpu)
            else:
                # Ensure models are on GPU for high VRAM mode
                self.text_encoder.to(gpu)
                self.text_encoder_2.to(gpu)
            
            llama_vec, clip_l_pooler = encode_prompt_conds(
                prompt, self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
            )
            
            # Ensure tensors are on the correct device
            llama_vec = llama_vec.to(gpu)
            clip_l_pooler = clip_l_pooler.to(gpu)
            
            # Negative prompt encoding
            if cfg == 1:
                llama_vec_n, clip_l_pooler_n = torch.zeros_like(llama_vec), torch.zeros_like(clip_l_pooler)
            else:
                llama_vec_n, clip_l_pooler_n = encode_prompt_conds(
                    "", self.text_encoder, self.text_encoder_2, self.tokenizer, self.tokenizer_2
                )
                # Ensure negative prompt tensors are on the correct device
                llama_vec_n = llama_vec_n.to(gpu)
                clip_l_pooler_n = clip_l_pooler_n.to(gpu)
            
            llama_vec, llama_attention_mask = crop_or_pad_yield_mask(llama_vec, length=512)
            llama_vec_n, llama_attention_mask_n = crop_or_pad_yield_mask(llama_vec_n, length=512)
            
            # Ensure all tensors are on GPU
            llama_vec = llama_vec.to(gpu)
            llama_vec_n = llama_vec_n.to(gpu)
            llama_attention_mask = llama_attention_mask.to(gpu)
            llama_attention_mask_n = llama_attention_mask_n.to(gpu)
            
            # Image processing
            job_manager.update_progress(job_id, 15.0, "Processing image...")
            
            H, W, C = input_image.shape
            height, width = find_nearest_bucket(H, W, resolution=640)
            input_image_np = resize_and_center_crop(input_image, target_width=width, target_height=height)
            
            # Save input image
            timestamp = generate_timestamp()
            input_filename = os.path.join(settings.OUTPUT_DIR, f'{job_id}_{timestamp}_input.png')
            Image.fromarray(input_image_np).save(input_filename)
            
            input_image_pt = torch.from_numpy(input_image_np).float() / 127.5 - 1
            input_image_pt = input_image_pt.permute(2, 0, 1)[None, :, None].to(gpu)
            
            # VAE encoding
            job_manager.update_progress(job_id, 20.0, "VAE encoding...")
            
            if not self.high_vram:
                load_model_as_complete(self.vae, target_device=gpu)
            else:
                self.vae.to(gpu)
            
            start_latent = vae_encode(input_image_pt, self.vae)
            start_latent = start_latent.to(gpu)
            
            # Clean up input image tensor
            del input_image_pt
            torch.cuda.empty_cache()
            
            # CLIP Vision encoding
            job_manager.update_progress(job_id, 25.0, "CLIP Vision encoding...")
            
            if not self.high_vram:
                load_model_as_complete(self.image_encoder, target_device=gpu)
            else:
                self.image_encoder.to(gpu)
            
            image_encoder_output = hf_clip_vision_encode(input_image_np, self.feature_extractor, self.image_encoder)
            image_encoder_last_hidden_state = image_encoder_output.last_hidden_state.to(gpu)
            
            # Clean up image encoder output
            del image_encoder_output
            torch.cuda.empty_cache()
            
            # Convert to correct dtypes
            llama_vec = llama_vec.to(transformer.dtype)
            llama_vec_n = llama_vec_n.to(transformer.dtype)
            clip_l_pooler = clip_l_pooler.to(transformer.dtype)
            clip_l_pooler_n = clip_l_pooler_n.to(transformer.dtype)
            image_encoder_last_hidden_state = image_encoder_last_hidden_state.to(transformer.dtype)
            
            # Start sampling
            job_manager.update_progress(job_id, 30.0, "Starting video generation...")
            
            rnd = torch.Generator("cpu").manual_seed(seed)
            num_frames = latent_window_size * 4 - 3
            
            # Initialize history
            if use_f1_model:
                history_latents = torch.zeros(size=(1, 16, 16 + 2 + 1, height // 8, width // 8), dtype=torch.float32).cpu()
                history_latents = torch.cat([history_latents, start_latent.to(history_latents)], dim=2)
                total_generated_latent_frames = 1
            else:
                history_latents = torch.zeros(size=(1, 16, 1 + 2 + 16, height // 8, width // 8), dtype=torch.float32).cpu()
                total_generated_latent_frames = 0
            
            history_pixels = None
            
            # Progress callback
            def progress_callback(d):
                current_step = d['i'] + 1
                section_progress = (current_step / steps) * 60  # 60% for generation
                total_progress = 30 + section_progress
                
                message = f'Sampling step {current_step}/{steps}'
                frames_generated = max(0, total_generated_latent_frames * 4 - 3)
                video_length = max(0, frames_generated / 30)
                
                job_manager.update_progress(
                    job_id, 
                    total_progress, 
                    message,
                    current_step=current_step,
                    generated_frames=frames_generated,
                    video_length=video_length
                )
            
            # Generation loop
            if use_f1_model:
                # F1 model generation logic
                for section_index in range(total_latent_sections):
                    if not self.high_vram:
                        unload_complete_models()
                        move_model_to_device_with_memory_preservation(
                            transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
                        )
                    
                    if use_teacache:
                        transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                    else:
                        transformer.initialize_teacache(enable_teacache=False)
                    
                    # F1 specific indices and latents setup
                    indices = torch.arange(0, sum([1, 16, 2, 1, latent_window_size])).unsqueeze(0)
                    clean_latent_indices_start, clean_latent_4x_indices, clean_latent_2x_indices, clean_latent_1x_indices, latent_indices = indices.split([1, 16, 2, 1, latent_window_size], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_start, clean_latent_1x_indices], dim=1)
                    
                    clean_latents_4x, clean_latents_2x, clean_latents_1x = history_latents[:, :, -sum([16, 2, 1]):, :, :].split([16, 2, 1], dim=2)
                    clean_latents = torch.cat([start_latent.to(history_latents), clean_latents_1x], dim=2)
                    
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=latent_window_size * 4 - 3,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=progress_callback,
                    )
                    
                    total_generated_latent_frames += int(generated_latents.shape[2])
                    history_latents = torch.cat([history_latents, generated_latents.to(history_latents)], dim=2)
                    
                    # Decode section
                    self._decode_section(job_id, history_latents, history_pixels, total_generated_latent_frames, 
                                       latent_window_size, False, timestamp, mp4_crf)
            
            else:
                # Standard model generation logic
                latent_paddings = reversed(range(total_latent_sections))
                if total_latent_sections > 4:
                    latent_paddings = [3] + [2] * (total_latent_sections - 3) + [1, 0]
                
                for latent_padding in latent_paddings:
                    is_last_section = latent_padding == 0
                    latent_padding_size = latent_padding * latent_window_size
                    
                    # Standard model indices and latents setup
                    indices = torch.arange(0, sum([1, latent_padding_size, latent_window_size, 1, 2, 16])).unsqueeze(0)
                    clean_latent_indices_pre, blank_indices, latent_indices, clean_latent_indices_post, clean_latent_2x_indices, clean_latent_4x_indices = indices.split([1, latent_padding_size, latent_window_size, 1, 2, 16], dim=1)
                    clean_latent_indices = torch.cat([clean_latent_indices_pre, clean_latent_indices_post], dim=1)
                    
                    clean_latents_pre = start_latent.to(history_latents)
                    clean_latents_post, clean_latents_2x, clean_latents_4x = history_latents[:, :, :1 + 2 + 16, :, :].split([1, 2, 16], dim=2)
                    clean_latents = torch.cat([clean_latents_pre, clean_latents_post], dim=2)
                    
                    if not self.high_vram:
                        unload_complete_models()
                        move_model_to_device_with_memory_preservation(
                            transformer, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
                        )
                    
                    if use_teacache:
                        transformer.initialize_teacache(enable_teacache=True, num_steps=steps)
                    else:
                        transformer.initialize_teacache(enable_teacache=False)
                    
                    generated_latents = sample_hunyuan(
                        transformer=transformer,
                        sampler='unipc',
                        width=width,
                        height=height,
                        frames=num_frames,
                        real_guidance_scale=cfg,
                        distilled_guidance_scale=gs,
                        guidance_rescale=rs,
                        num_inference_steps=steps,
                        generator=rnd,
                        prompt_embeds=llama_vec,
                        prompt_embeds_mask=llama_attention_mask,
                        prompt_poolers=clip_l_pooler,
                        negative_prompt_embeds=llama_vec_n,
                        negative_prompt_embeds_mask=llama_attention_mask_n,
                        negative_prompt_poolers=clip_l_pooler_n,
                        device=gpu,
                        dtype=torch.bfloat16,
                        image_embeddings=image_encoder_last_hidden_state,
                        latent_indices=latent_indices,
                        clean_latents=clean_latents,
                        clean_latent_indices=clean_latent_indices,
                        clean_latents_2x=clean_latents_2x,
                        clean_latent_2x_indices=clean_latent_2x_indices,
                        clean_latents_4x=clean_latents_4x,
                        clean_latent_4x_indices=clean_latent_4x_indices,
                        callback=progress_callback,
                    )
                    
                    if is_last_section:
                        generated_latents = torch.cat([start_latent.to(generated_latents), generated_latents], dim=2)
                    
                    total_generated_latent_frames += int(generated_latents.shape[2])
                    history_latents = torch.cat([generated_latents.to(history_latents), history_latents], dim=2)
                    
                    # Decode section
                    history_pixels = self._decode_section(job_id, history_latents, history_pixels, 
                                                        total_generated_latent_frames, latent_window_size, 
                                                        is_last_section, timestamp, mp4_crf)
                    
                    if is_last_section:
                        break
            
            # Final processing
            job_manager.update_progress(job_id, 95.0, "Finalizing video...")
            
            # Generate final output filename
            output_filename = os.path.join(settings.OUTPUT_DIR, f'{job_id}_{timestamp}_final.mp4')
            
            # Clean up GPU memory
            if not self.high_vram:
                unload_complete_models()
            
            # Complete job
            result_data = {
                "video_filename": output_filename,
                "input_filename": input_filename,
                "duration": duration,
                "frames_generated": max(0, total_generated_latent_frames * 4 - 3),
                "video_length": max(0, (total_generated_latent_frames * 4 - 3) / 30),
                "model_used": "FramePack-F1" if use_f1_model else "FramePack",
                "parameters": {
                    "prompt": prompt,
                    "mode": mode,
                    "seed": seed,
                    "steps": steps,
                    "cfg_scale": cfg,
                    "distilled_cfg_scale": gs,
                    "use_teacache": use_teacache
                }
            }
            
            job_manager.complete_job(job_id, result_data)
            
        except Exception as e:
            error_msg = f"Generation failed: {str(e)}"
            print(f"Job {job_id} failed: {error_msg}")
            traceback.print_exc()
            job_manager.fail_job(job_id, error_msg)
    
    def _decode_section(self, job_id, history_latents, history_pixels, total_generated_latent_frames,
                       latent_window_size, is_last_section, timestamp, mp4_crf):
        """Decode a section of latents to pixels with ultra-aggressive memory management"""
        # Clear GPU cache before VAE operations
        torch.cuda.empty_cache()
        
        # Always use aggressive memory management regardless of high_vram setting
        # Offload ALL models except VAE with maximum memory preservation
        offload_model_from_device_for_memory_preservation(self.transformer, target_device=gpu, preserved_memory_gb=15)
        offload_model_from_device_for_memory_preservation(self.transformer_f1, target_device=gpu, preserved_memory_gb=15)
        offload_model_from_device_for_memory_preservation(self.text_encoder, target_device=gpu, preserved_memory_gb=15)
        offload_model_from_device_for_memory_preservation(self.text_encoder_2, target_device=gpu, preserved_memory_gb=15)
        offload_model_from_device_for_memory_preservation(self.image_encoder, target_device=gpu, preserved_memory_gb=15)
        
        # Clear cache again after offloading
        torch.cuda.empty_cache()
        
        # Load VAE with memory check
        load_model_as_complete(self.vae, target_device=gpu)
        
        real_history_latents = history_latents[:, :, :total_generated_latent_frames, :, :]
        
        try:
            if history_pixels is None:
                # Always use ultra-conservative chunked processing for memory safety
                if real_history_latents.shape[2] > 8:  # If more than 8 frames, use chunking
                    # Start with very small chunks (2 frames)
                    chunk_size = 2
                    pixel_chunks = []
                    
                    for i in range(0, real_history_latents.shape[2], chunk_size):
                        end_idx = min(i + chunk_size, real_history_latents.shape[2])
                        chunk_latents = real_history_latents[:, :, i:end_idx, :, :]
                        
                        # Clear cache before each chunk
                        torch.cuda.empty_cache()
                        
                        chunk_pixels = vae_decode(chunk_latents, self.vae).cpu()
                        pixel_chunks.append(chunk_pixels)
                        
                        # Move chunk to CPU immediately
                        del chunk_pixels, chunk_latents
                        torch.cuda.empty_cache()
                    
                    # Concatenate chunks on CPU
                    history_pixels = torch.cat(pixel_chunks, dim=2)
                    del pixel_chunks
                else:
                    # For small sequences, process frame by frame for maximum safety
                    pixel_frames = []
                    for frame_idx in range(real_history_latents.shape[2]):
                        frame_latent = real_history_latents[:, :, frame_idx:frame_idx+1, :, :]
                        torch.cuda.empty_cache()
                        frame_pixel = vae_decode(frame_latent, self.vae).cpu()
                        pixel_frames.append(frame_pixel)
                        del frame_pixel, frame_latent
                        torch.cuda.empty_cache()
                    
                    history_pixels = torch.cat(pixel_frames, dim=2)
                    del pixel_frames
            else:
                section_latent_frames = (latent_window_size * 2 + 1) if is_last_section else (latent_window_size * 2)
                overlapped_frames = latent_window_size * 4 - 3
                
                section_latents = real_history_latents[:, :, :section_latent_frames]
                
                # Clear cache before decoding
                torch.cuda.empty_cache()
                
                current_pixels = vae_decode(section_latents, self.vae).cpu()
                
                # Fix overlap calculation - ensure overlap doesn't exceed current sequence length
                actual_overlap = min(overlapped_frames, current_pixels.shape[2] - 1)
                if actual_overlap <= 0:
                    # If no valid overlap, just concatenate
                    history_pixels = torch.cat([history_pixels, current_pixels], dim=2)
                else:
                    history_pixels = soft_append_bcthw(current_pixels, history_pixels, actual_overlap)
                
                # Clean up intermediate tensors
                del current_pixels, section_latents
                torch.cuda.empty_cache()
        
        except torch.cuda.OutOfMemoryError as e:
            print(f"⚠️ CUDA OOM during VAE decode, attempting emergency single-frame recovery: {e}")
            
            # Emergency memory cleanup
            torch.cuda.empty_cache()
            
            # Emergency fallback: Always process frame by frame
            print(f"🚨 Processing {real_history_latents.shape[2]} frames individually for maximum memory safety")
            pixel_frames = []
            
            for frame_idx in range(real_history_latents.shape[2]):
                try:
                    frame_latent = real_history_latents[:, :, frame_idx:frame_idx+1, :, :]
                    
                    # Clear cache before each frame
                    torch.cuda.empty_cache()
                    
                    frame_pixel = vae_decode(frame_latent, self.vae).cpu()
                    pixel_frames.append(frame_pixel)
                    
                    # Immediate cleanup
                    del frame_pixel, frame_latent
                    torch.cuda.empty_cache()
                    
                    if frame_idx % 5 == 0:  # Progress update every 5 frames
                        print(f"📹 Processed frame {frame_idx + 1}/{real_history_latents.shape[2]}")
                        
                except torch.cuda.OutOfMemoryError as frame_e:
                    print(f"❌ Failed to process frame {frame_idx}: {frame_e}")
                    # If even single frame fails, we need to abort
                    raise RuntimeError(f"CUDA OOM on single frame processing - insufficient GPU memory. Frame {frame_idx} failed: {frame_e}")
            
            if pixel_frames:
                history_pixels = torch.cat(pixel_frames, dim=2)
                del pixel_frames
                print(f"✅ Successfully recovered using single-frame processing")
            else:
                raise RuntimeError("No frames could be processed due to memory constraints")
        
        # Clean up latents
        del real_history_latents
        torch.cuda.empty_cache()
        
        # Always unload models after VAE operations for maximum memory recovery
        unload_complete_models()
        torch.cuda.empty_cache()
        
        # Save intermediate video
        output_filename = os.path.join(settings.OUTPUT_DIR, f'{job_id}_{timestamp}_{total_generated_latent_frames}.mp4')
        save_bcthw_as_mp4(history_pixels, output_filename, fps=30, crf=mp4_crf)
        
        return history_pixels

# Global worker instance
framepack_worker = FramePackWorker()

# Celery task wrapper
@celery.task(bind=True)
def process_job_task(self, job_id: str):
    """Celery task wrapper for video generation job processing"""
    return framepack_worker.process_job(job_id)