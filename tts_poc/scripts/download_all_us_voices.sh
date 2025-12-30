#!/bin/bash
cd piper_models

# Get all US English voice paths
voices=(
    "en/en_US/amy/low/en_US-amy-low"
    "en/en_US/amy/medium/en_US-amy-medium"
    "en/en_US/arctic/medium/en_US-arctic-medium"
    "en/en_US/bryce/medium/en_US-bryce-medium"
    "en/en_US/danny/low/en_US-danny-low"
    "en/en_US/hfc_female/medium/en_US-hfc_female-medium"
    "en/en_US/hfc_male/medium/en_US-hfc_male-medium"
    "en/en_US/joe/medium/en_US-joe-medium"
    "en/en_US/john/medium/en_US-john-medium"
    "en/en_US/kathleen/low/en_US-kathleen-low"
    "en/en_US/kristin/medium/en_US-kristin-medium"
    "en/en_US/kusal/medium/en_US-kusal-medium"
    "en/en_US/l2arctic/medium/en_US-l2arctic-medium"
    "en/en_US/lessac/high/en_US-lessac-high"
    "en/en_US/lessac/low/en_US-lessac-low"
    "en/en_US/lessac/medium/en_US-lessac-medium"
    "en/en_US/libritts/high/en_US-libritts-high"
    "en/en_US/libritts_r/medium/en_US-libritts_r-medium"
    "en/en_US/ljspeech/high/en_US-ljspeech-high"
    "en/en_US/ljspeech/medium/en_US-ljspeech-medium"
    "en/en_US/norman/medium/en_US-norman-medium"
    "en/en_US/ryan/high/en_US-ryan-high"
    "en/en_US/ryan/low/en_US-ryan-low"
    "en/en_US/ryan/medium/en_US-ryan-medium"
)

for voice in "${voices[@]}"; do
    filename=$(basename "$voice")
    echo "Downloading $filename..."
    curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/${voice}.onnx"
    curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/${voice}.onnx.json"
done

echo "Done! Downloaded ${#voices[@]} US English voices."