from TTS.api import TTS

# Load XTTS model (first run downloads ~2GB)
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cuda")

# Test texts with speed/pause simulation
texts = [
    "Welcome to our story.",  # Normal
    "Chapter one... The Journey Begins.",  # Pause via punctuation
    "The traveler walked slowly through the misty morning."  # Longer sentence
]

# Generate with default voice
for i, text in enumerate(texts):
    tts.tts_to_file(
        text=text,
        file_path=f"narration_{i}.wav",
        speaker_wav="path/to/reference_voice.wav",  # Optional: use your own voice sample
        language="en",
        speed=1.0  # Adjust: 0.5=slow, 1.5=fast
    )
    print(f"Generated audio {i}")

# Without reference voice (uses default)
tts.tts_to_file(
    text="This uses a default voice.",
    file_path="default_voice_test.wav",
    language="en"
)