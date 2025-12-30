#!/bin/bash
cd piper_models

# All GB English voices
voices=(
    "en/en_GB/alan/low/en_GB-alan-low"
    "en/en_GB/alan/medium/en_GB-alan-medium"
    "en/en_GB/alba/medium/en_GB-alba-medium"
    "en/en_GB/aru/medium/en_GB-aru-medium"
    "en/en_GB/cori/high/en_GB-cori-high"
    "en/en_GB/cori/medium/en_GB-cori-medium"
    "en/en_GB/jenny_dioco/medium/en_GB-jenny_dioco-medium"
    "en/en_GB/northern_english_male/medium/en_GB-northern_english_male-medium"
    "en/en_GB/semaine/medium/en_GB-semaine-medium"
    "en/en_GB/southern_english_female/low/en_GB-southern_english_female-low"
    "en/en_GB/vctk/medium/en_GB-vctk-medium"
)

for voice in "${voices[@]}"; do
    filename=$(basename "$voice")
    echo "Downloading $filename..."
    curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/${voice}.onnx"
    curl -L -O "https://huggingface.co/rhasspy/piper-voices/resolve/main/${voice}.onnx.json"
done

echo "Done! Downloaded ${#voices[@]} British English voices."