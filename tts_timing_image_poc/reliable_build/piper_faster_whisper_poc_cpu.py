import os
import sys

cudnn_path = os.path.join(sys.prefix, "lib/python3.11/site-packages/nvidia/cudnn/lib")
os.environ['LD_LIBRARY_PATH'] = f"{cudnn_path}:{os.environ.get('LD_LIBRARY_PATH', '')}"

import pandas as pd
from piper import PiperVoice
import wave
from faster_whisper import WhisperModel
import json
import os

# Setup
os.makedirs("audio", exist_ok=True)

# Load voice
voice = PiperVoice.load("../../piper_models/en_GB-alan-medium.onnx", use_cuda=False)
voice.config.length_scale = 0.8
voice.config.noise_scale = 1.0
voice.config.noise_w_scale = 1.0

# Load faster-whisper model
print("Loading Whisper model...")
whisper_model = WhisperModel("base", device="cpu", compute_type="int8")

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
    
    print(f"Line {i}: Getting word timestamps...")
    
    # Transcribe with word-level timestamps
    segments, info = whisper_model.transcribe(audio_path, word_timestamps=True)
    
    # Extract word timings
    timings = {"words": []}
    for segment in segments:
        if segment.words:
            for word in segment.words:
                timings["words"].append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
    
    # Save timings
    with open(timings_path, "w") as f:
        json.dump(timings, f, indent=2)
    
    print(f"Line {i}: Saved audio + timings")

print("\nPhase 1 Complete! All audio and timings generated.")