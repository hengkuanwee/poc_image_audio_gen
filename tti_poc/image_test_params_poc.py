from diffusers import StableDiffusionXLPipeline
import torch
import time

# Load model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Test prompt
prompt = "Vibrant swirling impasto abstract painting, a person sitting alone in contemplation, warm lighting, introspective mood"

# Negative prompt (things to avoid)
negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text"

# Test different configurations
configs = [
    {"name": "fast", "steps": 20, "guidance": 7.0},
    {"name": "balanced", "steps": 30, "guidance": 7.5},
    {"name": "quality", "steps": 50, "guidance": 8.0},
    {"name": "high_guidance", "steps": 30, "guidance": 12.0},
    {"name": "low_guidance", "steps": 30, "guidance": 5.0},
]

seed = 42

for config in configs:
    generator = torch.Generator("cuda").manual_seed(seed)
    
    print(f"\nGenerating: {config['name']}")
    print(f"  Steps: {config['steps']}, Guidance: {config['guidance']}")
    
    start_time = time.time()
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1920,
        height=1080,
        num_inference_steps=config['steps'],
        guidance_scale=config['guidance'],
        generator=generator
    ).images[0]
    
    elapsed = time.time() - start_time
    
    image.save(f"params_{config['name']}.png")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved: params_{config['name']}.png")

print("\n\nDone! Compare images:")
print("- 'fast' vs 'quality': Does more steps improve quality?")
print("- 'low_guidance' vs 'high_guidance': Which follows prompt better?")
print("- Check generation times - what's your speed/quality sweet spot?")