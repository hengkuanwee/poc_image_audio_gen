import pandas as pd
import json
from moviepy.editor import ImageClip, AudioFileClip, CompositeVideoClip, TextClip, VideoFileClip, concatenate_videoclips
# from moviepy import ImageClip, AudioFileClip, CompositeVideoClip, TextClip
import os

# Setup
os.makedirs("clips", exist_ok=True)

# Subtitle settings - UPDATE THESE
MAX_SUBTITLE_WIDTH = 960
FONT_SIZE = 120  # Increased from 60
FONT = "Impact"
TEXT_COLOR = "white"
HIGHLIGHT_COLOR = "yellow"
STROKE_COLOR = "black"
STROKE_WIDTH = 3  # Increased from 3

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
    temp_clips = []
    
    # First pass: create all clips to measure total width
    for i, word in enumerate(words_text):
        color = HIGHLIGHT_COLOR if i == highlight_word_index else TEXT_COLOR
        
        txt_clip = TextClip(
            word,
            fontsize=FONT_SIZE,
            color=color,
            font=FONT,
            stroke_color=STROKE_COLOR,
            stroke_width=STROKE_WIDTH,
            method='caption',
            align='center',  # Add this
            size=(None, None)  # Add this - let it auto-size
        )
        temp_clips.append(txt_clip)
    
    # Calculate total width with spacing
    total_width = sum(clip.w for clip in temp_clips) + (len(temp_clips) - 1) * 20
    
    # Center the entire subtitle block
    x_start = (video_width - total_width) / 2
    x_position = x_start
    
    # Position each word
    for clip in temp_clips:
        positioned_clip = clip.set_position((x_position, video_height - 150))
        word_clips.append(positioned_clip)
        x_position += clip.w + 20
    
    return word_clips

# Read CSV
df = pd.read_csv("script_prompts.csv")

# # Phase 3: Create clips for each line
# for i, row in df.iterrows():
#     print(f"\nLine {i}: Creating clip...")
    
#     # Load assets
#     audio_path = f"audio/chapter1_line{i}.wav"
#     image_path = f"images/chapter1_line{i}.png"
#     timings_path = f"audio/chapter1_line{i}_timings.json"
    
#     with open(timings_path, 'r') as f:
#         timings = json.load(f)
    
#     words = timings['words']
    
#     # Load audio to get duration
#     audio_clip = AudioFileClip(audio_path)
#     duration = audio_clip.duration
    
#     # Create base video (image for entire duration)
#     image_clip = ImageClip(image_path).set_duration(duration)
    
#     # Group words into chunks
#     chunks = create_subtitle_chunks(words, MAX_SUBTITLE_WIDTH, FONT_SIZE)
    
#     # Create subtitle clips for each chunk
#     all_subtitle_clips = []

#     for chunk_idx, chunk in enumerate(chunks):
#         chunk_start = chunk[0]['start']
#         chunk_end = chunk[-1]['end']
        
#         # Extend chunk display slightly to avoid gaps
#         if chunk_idx < len(chunks) - 1:
#             next_chunk_start = chunks[chunk_idx + 1][0]['start']
#             chunk_end = next_chunk_start  # Display until next chunk starts
        
#         # For each word in chunk, create highlighted version
#         for word_idx, word_data in enumerate(chunk):
#             word_start = word_data['start']
#             word_end = word_data['end']
            
#             # Extend word highlight to next word or chunk end
#             if word_idx < len(chunk) - 1:
#                 word_end = chunk[word_idx + 1]['start']
#             else:
#                 word_end = chunk_end
            
#             # Create subtitle with this word highlighted
#             subtitle_clips = create_subtitle_clip(
#                 chunk, 
#                 1920, 
#                 1080, 
#                 highlight_word_index=word_idx
#             )
            
#             # Set timing for all words in this subtitle
#             for clip in subtitle_clips:
#                 clip = clip.set_start(word_start).set_duration(word_end - word_start)
#                 all_subtitle_clips.append(clip)
    
#     # Composite everything
#     final_clip = CompositeVideoClip(
#         [image_clip] + all_subtitle_clips,
#         size=(1920, 1080)
#     ).set_audio(audio_clip)
    
#     # Export
#     output_path = f"clips/chapter1_line{i}.mp4"
#     final_clip.write_videofile(
#         output_path,
#         fps=30,
#         codec='libx264',
#         audio_codec='aac'
#     )
    
#     print(f"Line {i}: Saved to {output_path}")

# print("\nPhase 3 Complete! All clips created.")


print("\n" + "="*50)
print("Phase 4: Concatenating all clips into final video...")
print("="*50)

# Load all clips
all_clips = []
for i in range(len(df)):
    clip_path = f"clips/chapter1_line{i}.mp4"
    if os.path.exists(clip_path):
        print(f"Loading clip {i}...")
        clip = VideoFileClip(clip_path)
        all_clips.append(clip)

# Concatenate clips
print("\nConcatenating clips...")
final_video = concatenate_videoclips(all_clips, method="compose")

# Load film grain overlay
print("\nAdding film grain overlay...")
film_grain = VideoFileClip("input/film_grain_overlay.mp4")

# Manual resize function
def resize_frame(get_frame, t):
    import cv2
    frame = get_frame(t)
    return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

film_grain = film_grain.fl(resize_frame)

# Loop the grain if video is longer
if final_video.duration > film_grain.duration:
    film_grain = film_grain.loop(duration=final_video.duration)
else:
    film_grain = film_grain.subclip(0, final_video.duration)

# Apply screen blend: removes blacks, keeps bright pixels
def screen_blend(get_frame, t):
    import numpy as np
    grain_frame = get_frame(t).astype(float) / 255.0
    
    # Amplify the grain visibility (since max is only ~100/255)
    grain_frame = np.power(grain_frame, 0.5) * 1.5  # Brighten
    grain_frame = np.clip(grain_frame, 0, 1)
    
    return (grain_frame * 255).astype('uint8')

film_grain = film_grain.fl(screen_blend)

# Use screen opacity for blending (not regular opacity)
film_grain = film_grain.set_opacity(0.3)

# Composite
final_with_grain = CompositeVideoClip([final_video, film_grain])

# Export final video
output_path = "output/chapter1_final.mp4"
os.makedirs("output", exist_ok=True)

print(f"\nExporting final video to {output_path}...")
final_with_grain.write_videofile(
    output_path,
    fps=30,
    codec='libx264',
    audio_codec='aac'
)

print(f"\n✅ Complete! Final video saved to: {output_path}")