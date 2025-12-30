from diffusers import StableDiffusionXLPipeline
import torch

# Load model
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Test prompt
prompt = "oil painting style, a person sitting alone in contemplation, warm lighting, introspective mood"

# Test different dimensions
dimensions = [
    ("youtube_standard", 1920, 1080),   # 16:9 landscape
    ("youtube_shorts", 1080, 1920),     # 9:16 vertical
    ("square", 1024, 1024),             # 1:1 (Instagram/thumbnail)
]

seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

for name, width, height in dimensions:
    print(f"Generating {name} ({width}x{height})...")
    
    image = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=30,
        generator=generator
    ).images[0]
    
    image.save(f"test_{name}_{width}x{height}.png")
    print(f"Saved: test_{name}_{width}x{height}.png")

print("\nDone! Check image quality and generation time for each dimension.")