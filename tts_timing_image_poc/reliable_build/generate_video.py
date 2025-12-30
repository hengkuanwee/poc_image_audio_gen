import pandas as pd
import json
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
# from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
import os

# Setup
os.makedirs("clips", exist_ok=True)

# Subtitle settings
MAX_SUBTITLE_WIDTH = 960  # 50% of 1920px
FONT_SIZE = 60
FONT = "Arial-Bold"
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "yellow"
STROKE_COLOR = "black"
STROKE_WIDTH = 3

def create_subtitle_chunks(words, max_width_px, font_size):
    """Group words into chunks that fit within max width"""
    # Rough estimate: average char width ~ font_size * 0.6
    avg_char_width = font_size * 0.6
    chunks = []
    current_chunk = []
    current_width = 0
    
    for word_data in words:
        word = word_data['word'].strip()
        word_width = len(word) * avg_char_width + avg_char_width  # +space
        
        if current_width + word_width > max_width_px and current_chunk:
            # Start new chunk
            chunks.append(current_chunk)
            current_chunk = [word_data]
            current_width = word_width
        else:
            current_chunk.append(word_data)
            current_width += word_width
    
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

def create_subtitle_clip(chunk, video_width, video_height, highlight_word_index):
    """Create a single subtitle line with one word highlighted"""
    words_text = [w['word'].strip() for w in chunk]
    
    # Create text clips for each word
    word_clips = []
    x_position = (video_width - MAX_SUBTITLE_WIDTH) / 2  # Start position
    
    for i, word in enumerate(words_text):
        color = HIGHLIGHT_COLOR if i == highlight_word_index else TEXT_COLOR
        
        txt_clip = TextClip(
            word,
            fontsize=FONT_SIZE,
            color=color,
            font=FONT,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method='caption'
        ).set_position((x_position, video_height - 150))
        
        word_clips.append(txt_clip)
        x_position += txt_clip.w + 20  # 20px spacing between words
    
    return word_clips

# Read CSV
df = pd.read_csv("script_prompts.csv")

# Phase 3: Create clips for each line
for i, row in df.iterrows():
    print(f"\nLine {i}: Creating clip...")
    
    # Load assets
    audio_path = f"audio/chapter1_line{i}.wav"
    image_path = f"images/chapter1_line{i}.png"
    timings_path = f"audio/chapter1_line{i}_timings.json"
    
    with open(timings_path, 'r') as f:
        timings = json.load(f)
    
    words = timings['words']
    
    # Load audio to get duration
    audio_clip = AudioFileClip(audio_path)
    duration = audio_clip.duration
    
    # Create base video (image for entire duration)
    image_clip = ImageClip(image_path).set_duration(duration)
    
    # Group words into chunks
    chunks = create_subtitle_chunks(words, MAX_SUBTITLE_WIDTH, FONT_SIZE)
    
    # Create subtitle clips for each chunk
    all_subtitle_clips = []
    
    for chunk_idx, chunk in enumerate(chunks):
        chunk_start = chunk[0]['start']
        chunk_end = chunk[-1]['end']
        
        # For each word in chunk, create highlighted version
        for word_idx, word_data in enumerate(chunk):
            word_start = word_data['start']
            word_end = word_data['end']
            
            # Create subtitle with this word highlighted
            subtitle_clips = create_subtitle_clip(
                chunk, 
                1920, 
                1080, 
                highlight_word_index=word_idx
            )
            
            # Set timing for all words in this subtitle
            for clip in subtitle_clips:
                clip = clip.set_start(word_start).set_duration(word_end - word_start)
                all_subtitle_clips.append(clip)
    
    # Composite everything
    final_clip = CompositeVideoClip(
        [image_clip] + all_subtitle_clips,
        size=(1920, 1080)
    ).set_audio(audio_clip)
    
    # Export
    output_path = f"clips/chapter1_line{i}.mp4"
    final_clip.write_videofile(
        output_path,
        fps=30,
        codec='libx264',
        audio_codec='aac'
    )
    
    print(f"Line {i}: Saved to {output_path}")

print("\nPhase 3 Complete! All clips created.")