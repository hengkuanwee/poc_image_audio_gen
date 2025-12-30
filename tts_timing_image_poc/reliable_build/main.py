import pandas as pd
from piper import PiperVoice
import wave
import whisperx
import json
import os

# Setup
os.makedirs("audio", exist_ok=True)

# Load voice
voice = PiperVoice.load("../piper_models/en_GB-alan-medium.onnx", use_cuda=False)
voice.config.length_scale = 0.8
voice.config.noise_scale = 1.0
voice.config.noise_w_scale = 1.0

# Load WhisperX model (one-time, will download ~1GB on first run)
print("Loading WhisperX model...")
whisper_model = whisperx.load_model("base", device="cuda", compute_type="float16")

# Read CSV
df = pd.read_csv("script_prompts.csv")

# Phase 1: Generate audio + timings for each line
for i, row in df.iterrows():
    script = row['script']
    
    audio_path = f"audio/chapter1_line{i}.wav"
    timings_path = f"audio/chapter1_line{i}_timings.json"
    
    # Skip if already generated
    if os.path.exists(audio_path) and os.path.exists(timings_path):
        print(f"Line {i}: Skipping (already exists)")
        continue
    
    print(f"Line {i}: Generating audio...")
    
    # Generate TTS
    audio_bytes = b""
    for chunk in voice.synthesize(script):
        audio_bytes += chunk.audio_int16_bytes
    
    # Save audio
    with wave.open(audio_path, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.writeframes(audio_bytes)
    
    print(f"Line {i}: Running forced alignment...")
    
    # Transcribe + align with WhisperX
    audio = whisperx.load_audio(audio_path)
    result = whisper_model.transcribe(audio, batch_size=16)
    
    # Align to get word-level timestamps
    align_model, metadata = whisperx.load_align_model(language_code="en", device="cuda")
    result_aligned = whisperx.align(
        result["segments"], 
        align_model, 
        metadata, 
        audio, 
        device="cuda"
    )
    
    # Save timings
    with open(timings_path, "w") as f:
        json.dump(result_aligned, f, indent=2)
    
    print(f"Line {i}: Saved audio + timings")

print("\nPhase 1 Complete! All audio and timings generated.")