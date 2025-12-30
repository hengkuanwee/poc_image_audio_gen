from moviepy.editor import VideoFileClip
import numpy as np

# Load overlay
overlay = VideoFileClip("input/film_grain_overlay.mp4")

print(f"Duration: {overlay.duration}s")
print(f"Size: {overlay.size}")
print(f"FPS: {overlay.fps}")

# Sample a few frames to check pixel values
print("\nChecking pixel values...")
for t in [0, 1, 2]:
    frame = overlay.get_frame(t)
    
    # Get min/max/mean values
    print(f"\nFrame at {t}s:")
    print(f"  Min RGB: {frame.min(axis=(0,1))}")
    print(f"  Max RGB: {frame.max(axis=(0,1))}")
    print(f"  Mean RGB: {frame.mean(axis=(0,1)).astype(int)}")
    
    # Check how many pixels are "black" (< 10)
    dark_pixels = np.sum(frame.max(axis=2) < 10)
    total_pixels = frame.shape[0] * frame.shape[1]
    dark_percent = (dark_pixels / total_pixels) * 100
    print(f"  Dark pixels (< 10): {dark_percent:.1f}%")