from diffusers import StableDiffusionXLPipeline
import torch

# Load model (first run downloads ~7GB)
pipe = StableDiffusionXLPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
)
pipe.to("cuda")

# Style keywords for consistency
style_prefix = "oil painting, impressionist style, thick brushstrokes,"

# Test prompts (story example)
prompts = [
    f"{style_prefix} a quiet village at dawn",
    f"{style_prefix} a traveler walking through the village",
    f"{style_prefix} the traveler entering a mysterious forest"
]

# Generate with same seed for style consistency
seed = 42
generator = torch.Generator("cuda").manual_seed(seed)

for i, prompt in enumerate(prompts):
    image = pipe(
        prompt=prompt,
        generator=generator,
        num_inference_steps=30
    ).images[0]
    
    image.save(f"story_frame_{i}.png")
    print(f"Generated frame {i}")