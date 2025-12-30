from piper import PiperVoice
import wave
import re

# Load script
with open("./tts_poc/text/narration_script.txt", "r") as f:
    script = f.read()

# Your two voices to test
voices_to_test = [
    "en_GB-alba-medium",  # Replace with your chosen voice 1
    "en_GB-semaine-medium",  # Replace with your chosen voice 2
]

for voice_name in voices_to_test:
    print(f"\nGenerating with {voice_name}...")
    
    # Load voice
    voice = PiperVoice.load(f"./piper_models/{voice_name}.onnx", use_cuda=False)
    
    # Apply config
    voice.config.length_scale = 0.8
    voice.config.noise_scale = 1.0
    voice.config.noise_w_scale = 1.0
    
    # Parse script for pauses
    segments = re.split(r'<pause:([\d.]+)>', script)
    
    audio_bytes = b""
    
    for i, segment in enumerate(segments):
        if i % 2 == 0:  # Text segment
            if segment.strip():
                # Generate speech
                for chunk in voice.synthesize(segment.strip()):
                    audio_bytes += chunk.audio_int16_bytes
        else:  # Pause duration
            pause_duration = float(segment)
            silence_samples = int(voice.config.sample_rate * pause_duration)
            silence = b'\x00\x00' * silence_samples
            audio_bytes += silence
    
    # Save
    output_file = f"full_narration_{voice_name}.wav"
    with wave.open(output_file, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(voice.config.sample_rate)
        wav_file.writeframes(audio_bytes)
    
    print(f"Saved: {output_file}")

print("\nDone! Compare the two narrations and pick your favorite.")