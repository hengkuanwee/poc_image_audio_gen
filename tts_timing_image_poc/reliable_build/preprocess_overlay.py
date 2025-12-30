from moviepy.editor import VideoFileClip, ImageClip, CompositeVideoClip
import numpy as np
import cv2

import os
import shutil
import moviepy.config as movieconf
# os.environ["FFMPEG_BINARY"] = "/usr/bin/ffmpeg"
movieconf.FFMPEG_BINARY = shutil.which("ffmpeg")

print("="*50)
print("Preprocessing Film Grain Overlay")
print("="*50)

# Step 1: Create TV border mask
print("\n1. Creating TV border with rounded edges...")

def create_tv_border(width, height):
    """Create vintage TV border with rounded corners"""
    # Start with white (transparent center)
    border = np.ones((height, width, 3), dtype=np.uint8) * 255
    
    # Define border thickness
    border_thickness = 20  # pixels
    corner_radius = 50  # pixels for rounded corners
    
    # Define border area (draw black on edges)
    # Top border
    cv2.rectangle(border, (0, 0), (width, border_thickness), (0, 0, 0), -1)
    # Bottom border
    cv2.rectangle(border, (0, height - border_thickness), (width, height), (0, 0, 0), -1)
    # Left border
    cv2.rectangle(border, (0, 0), (border_thickness, height), (0, 0, 0), -1)
    # Right border
    cv2.rectangle(border, (width - border_thickness, 0), (width, height), (0, 0, 0), -1)
    
    # Add rounded corners by drawing white circles on the inner corners
    x1, y1 = border_thickness, border_thickness
    x2, y2 = width - border_thickness, height - border_thickness
    
    # Fill corners with black first
    cv2.rectangle(border, (0, 0), (x1, y1), (0, 0, 0), -1)
    cv2.rectangle(border, (x2, 0), (width, y1), (0, 0, 0), -1)
    cv2.rectangle(border, (0, y2), (x1, height), (0, 0, 0), -1)
    cv2.rectangle(border, (x2, y2), (width, height), (0, 0, 0), -1)
    
    # Cut rounded corners with white circles
    cv2.circle(border, (x1, y1), corner_radius, (255, 255, 255), -1)
    cv2.circle(border, (x2, y1), corner_radius, (255, 255, 255), -1)
    cv2.circle(border, (x1, y2), corner_radius, (255, 255, 255), -1)
    cv2.circle(border, (x2, y2), corner_radius, (255, 255, 255), -1)
    
    return border

border_mask = create_tv_border(1920, 1080)
cv2.imwrite("tv_border.png", border_mask)
print("   ✓ TV border created and saved as tv_border.png")

# Step 2: Load and process film grain
print("\n2. Loading film grain overlay...")
overlay = VideoFileClip("input/film_grain_overlay.mp4")

# Step 3: Resize to 1920x1080
print("\n3. Resizing overlay to 1920x1080...")

def resize_frame(get_frame, t):
    frame = get_frame(t)
    return cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_LINEAR)

overlay = overlay.fl(resize_frame)

# Step 4: Process grain - brighten and add border in one pass
print("\n4. Processing grain and adding border...")

border_static = cv2.imread("tv_border.png")

def process_grain_with_border(get_frame, t):
    grain_frame = get_frame(t).astype(float) / 255.0
    # Amplify grain visibility
    grain_frame = np.power(grain_frame, 0.5) * 1.5
    grain_frame = np.clip(grain_frame, 0, 1)
    grain_frame = (grain_frame * 255).astype('uint8')
    
    # Apply border: where border is black (0), use black; otherwise keep grain
    border_mask = (border_static == 0).any(axis=2)  # Black pixels in border
    grain_frame[border_mask] = 0  # Set border areas to black
    
    return grain_frame

overlay_processed = overlay.fl(process_grain_with_border)

# Step 5: Export preprocessed overlay
print("\n5. Exporting preprocessed overlay...")
overlay_processed.write_videofile(
    "film_grain_final.mp4",
    codec='h264_nvenc',  # Direct NVENC codec
    fps=overlay.fps,
    bitrate="8000k",
    audio=False,
    ffmpeg_params=['-preset', 'fast']  # Just pass preset/gpu params
)

print("\n" + "="*50)
print("✓ Preprocessing complete!")
print("Use 'film_grain_final.mp4' in your main video script")
print("="*50)