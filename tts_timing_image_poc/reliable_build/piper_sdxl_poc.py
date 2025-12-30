import pandas as pd
from diffusers import StableDiffusionXLPipeline
import torch
import os

# Setup
os.makedirs("images", exist_ok=True)

# Load SDXL model (one-time, ~7GB download on first run)
print("Loading SDXL model...")
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Consistent settings for all images
STYLE_PREFIX = "cinematic oil painting style, dramatic lighting, muted color palette, film grain, highly detailed, masterpiece, 4k quality, "

NEGATIVE_PROMPT = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, words, letters, signature, oversaturated colors, cartoon, anime, 3d render, unrealistic, deformed, extra limbs, duplicate, disfigured"

num_steps = 40  # Increase for better quality
guidance_scale = 8.0

# Read CSV
df = pd.read_csv("script_prompts.csv")

# Phase 2: Generate images for each line
for i, row in df.iterrows():
    base_prompt = row['image_prompt']
    
    # Add style prefix
    full_prompt = STYLE_PREFIX + base_prompt
    
    image_path = f"images/chapter1_line{i}.png"
    
    if os.path.exists(image_path):
        print(f"Line {i}: Skipping image (already exists)")
        continue
    
    print(f"Line {i}: Generating image...")
    print(f"  Prompt: {full_prompt}")
    
    seed = 42 + i
    generator = torch.Generator("cuda").manual_seed(seed)
    
    image = pipe(
        prompt=full_prompt,
        negative_prompt=NEGATIVE_PROMPT,  # Add this
        width=1920,
        height=1080,
        num_inference_steps=num_steps,
        guidance_scale=guidance_scale,
        generator=generator
    ).images[0]
    
    image.save(image_path)
    print(f"Line {i}: Saved to {image_path}")

print("\nPhase 2 Complete! All images generated.")