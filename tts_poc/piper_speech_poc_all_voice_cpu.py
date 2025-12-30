from piper import PiperVoice
import wave
import os

# Your narration segments
segments = [
    "Let me ask you something... that most people spend their whole lives avoiding.",
    "When was the last time you actually sat with your pain?",
    "Not escaped it. Not numbed it. Not disguised it as productivity.",
    "But TRULY... FELT it?"
]

# Get all voices
voice_files = [f for f in os.listdir("./piper_models") if f.endswith(".onnx")]

print(f"Generating narration with {len(voice_files)} voices...\n")

for voice_file in voice_files:
    voice_name = voice_file.replace(".onnx", "")
    print(f"Generating: {voice_name}")
    
    # Load voice
    voice = PiperVoice.load(f"./piper_models/{voice_file}", use_cuda=False)
    voice.config.length_scale = 0.8
    voice.config.noise_scale = 1.0
    voice.config.noise_w_scale = 1.0
    # Generate with pauses
    audio_bytes = b""
    silence_duration = 0.8  # seconds
    
    for segment in segments:
        # Generate speech
        for chunk in voice.synthesize(segment):
            audio_bytes += chunk.audio_int16_bytes
        
        # Add silence between segments
        silence_samples = int(voice.config.sample_rate * silence_duration)
        silence = b'\x00\x00' * silence_samples
        audio_bytes += silence
    
    # Save
    with wave.open(f"narration_{voice_name}.wav", "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.writeframes(audio_bytes)

print("\nDone! Listen to all 'narration_*.wav' files and pick your favorite.")