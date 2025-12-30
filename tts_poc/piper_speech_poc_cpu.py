from piper import PiperVoice
import wave

# Load voice
voice = PiperVoice.load(
    "./piper_models/en_US-lessac-medium.onnx",
    use_cuda=False
)

# Test texts
texts = [
    "Welcome to our story.",
    "Chapter one. The Journey Begins.",
    "The traveler walked slowly through the misty morning."
]

# Generate audio
for i, text in enumerate(texts):
    # Collect all audio chunks
    audio_bytes = b""
    for chunk in voice.synthesize(text):
        audio_bytes += chunk.audio_int16_bytes  # This is the correct attribute!
    
    # Write WAV file
    with wave.open(f"piper_narration_{i}.wav", "wb") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.writeframes(audio_bytes)
    
    print(f"Generated audio {i}")

print("Done! Audio files created.")