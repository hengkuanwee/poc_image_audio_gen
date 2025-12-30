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
prompts = [
    "Cinematic oil painting of a lone figure sitting in dim candlelight, shadows swirling like emotions, hands resting in stillness, deep introspective atmosphere, dramatic Caravaggio-style chiaroscuro, rich texture, melancholic mood",

    "Medieval illuminated-manuscript scene where pain is personified as a gentle guide approaching a hesitant traveler, gold leaf accents, parchment texture, symbolic allegory, soft mystical glow, cinematic framing",

    "Epic Renaissance-style oil painting tableau of symbolic philosophers on distant cliffs—Stoics, Nietzsche, and Buddhist monks—connected by a glowing thread of insight, atmospheric clouds, dramatic lighting, wide cinematic composition",

    "Cinematic medieval-romantic oil painting of a figure walking through a raging storm, dark skies splitting to reveal soft golden light, symbolizing transformation through adversity, sweeping brushstrokes, dramatic wide shot",

    "Moody Baroque-style oil painting of a shattered mirror in a dim hall, each shard reflecting different emotions, soft light emerging through cracks, symbolic storytelling, rich contrast, cinematic dramatic lighting",

    "Epic mythic-medieval painting of a human form made of fragmented stone being reforged with glowing gold seams (kintsugi-inspired), radiant highlights, deep blues, symbolic rebirth, sweeping cinematic composition"
]

# Negative prompt (things to avoid)
# negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark, text"
negative_prompt = (
    "blurry, low resolution, grainy, distorted anatomy, malformed hands, extra limbs, extra fingers, "
    "cartoonish style, plastic textures, modern objects, modern clothing, sci-fi elements, washed out colors, "
    "overexposed, underexposed, flat lighting, digital artifacts, noisy, messy composition, text, watermark, "
    "signature, cropped, bad proportions, unnatural poses"
)

# Test different configurations
configs = [
    {"name": "fast", "steps": 20, "guidance": 7.0},
    {"name": "balanced", "steps": 30, "guidance": 7.5},
    {"name": "quality", "steps": 50, "guidance": 8.0},
    {"name": "high_guidance", "steps": 30, "guidance": 12.0},
    {"name": "low_guidance", "steps": 30, "guidance": 5.0},
]

seed = 42

for i, prompt in enumerate(prompts):
    generator = torch.Generator("cuda").manual_seed(seed)
    
    # print(f"\nGenerating: {config['name']}")
    # print(f"  Steps: {config['steps']}, Guidance: {config['guidance']}")
    
    start_time = time.time()
    
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=1920,
        height=1080,
        num_inference_steps=30,
        guidance_scale=7.5,
        generator=generator
    ).images[0]
    
    elapsed = time.time() - start_time
    
    image.save(f"{i}.png")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Saved: {i}.png")

# print("\n\nDone! Compare images:")
# print("- 'fast' vs 'quality': Does more steps improve quality?")
# print("- 'low_guidance' vs 'high_guidance': Which follows prompt better?")
# print("- Check generation times - what's your speed/quality sweet spot?")