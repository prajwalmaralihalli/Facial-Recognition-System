"""
media_loader.py — Visual Feedback Panel Loader
-----------------------------------------------
Handles loading static images (.jpg/.png) and animated GIFs for the
right-side display panel of the Facial Recognition System.

Supports:
  - Static images (JPEG, PNG) via OpenCV
  - Animated GIFs — all frames extracted into a list for cycling
  - Aspect-ratio-preserving resize with center-crop or letterbox padding
  - Blank fallback frame when files are missing or unreadable

Usage:
    from media_loader import load_image_panel, get_current_frame, resize_panel, make_blank_frame
"""

import cv2
import numpy as np
from pathlib import Path

# Default images directory (relative to this file's location)
IMAGES_DIR = Path(__file__).parent.parent / "images"


# ═════════════════════════════════════════════════════════════════════════════
# Blank / Fallback Frame
# ═════════════════════════════════════════════════════════════════════════════

def make_blank_frame(height: int = 480, width: int = 640,
                     text: str = "No image loaded") -> np.ndarray:
    """
    Create a black frame with centered gray text.

    Used as a fallback when an image or GIF is missing or fails to load.

    Args:
        height: Frame height in pixels.
        width:  Frame width in pixels.
        text:   Message to display in the center.

    Returns:
        A (height × width × 3) BGR numpy array.
    """
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    font  = cv2.FONT_HERSHEY_SIMPLEX
    scale, thickness, color = 0.7, 2, (120, 120, 120)
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    x = (width  - tw) // 2
    y = (height + th) // 2
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return frame


# ═════════════════════════════════════════════════════════════════════════════
# Static Image Loading (.jpg, .png)
# ═════════════════════════════════════════════════════════════════════════════

def load_image(filename: str, images_dir: str = None) -> np.ndarray | None:
    """
    Load a static image (.jpg or .png) from the images folder.

    Args:
        filename:   Name of the image file (e.g. "expression_smile.jpg").
        images_dir: Optional override for the images directory path.

    Returns:
        The image as a BGR numpy array, or None if the file is
        missing or unreadable.
    """
    folder   = Path(images_dir) if images_dir else IMAGES_DIR
    filepath = folder / filename

    if not filepath.exists():
        print(f"[media_loader] WARNING: File not found — {filepath}")
        return None

    img = cv2.imread(str(filepath))
    if img is None:
        print(f"[media_loader] WARNING: Could not decode — {filepath}")
        return None

    return img


# ═════════════════════════════════════════════════════════════════════════════
# Animated GIF Loading — Extract all frames into a list
# ═════════════════════════════════════════════════════════════════════════════

def load_gif_frames(filename: str, images_dir: str = None) -> list[np.ndarray]:
    """
    Load every frame of an animated GIF into a list of BGR numpy arrays.

    Uses OpenCV's VideoCapture which can open GIF files and read them
    frame-by-frame until ret=False.

    Args:
        filename:   Name of the GIF file (e.g. "reaction.gif").
        images_dir: Optional override for the images directory path.

    Returns:
        A list of frames (BGR np.ndarray). Returns an empty list if the
        file is missing or has zero readable frames.
    """
    folder   = Path(images_dir) if images_dir else IMAGES_DIR
    filepath = folder / filename

    if not filepath.exists():
        print(f"[media_loader] WARNING: GIF not found — {filepath}")
        return []

    cap = cv2.VideoCapture(str(filepath))
    if not cap.isOpened():
        print(f"[media_loader] WARNING: Could not open GIF — {filepath}")
        return []

    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()

    if not frames:
        print(f"[media_loader] WARNING: GIF has 0 readable frames — {filepath}")
    else:
        print(f"[media_loader] Loaded {len(frames)} frames from {filename}")

    return frames


# ═════════════════════════════════════════════════════════════════════════════
# GIF Frame Cycling
# ═════════════════════════════════════════════════════════════════════════════

def get_current_frame(frames: list[np.ndarray], frame_index: int) -> tuple[np.ndarray, int]:
    """
    Return the current GIF frame and the next frame index (wraps around).

    Args:
        frames:      List of BGR frames from load_gif_frames().
        frame_index: Current position in the frame list.

    Returns:
        (frame, next_index): The frame to display and the updated index.
        Returns (None, 0) if the frame list is empty.
    """
    if not frames:
        return (None, 0)

    idx = frame_index % len(frames)
    return (frames[idx].copy(), (idx + 1) % len(frames))


# ═════════════════════════════════════════════════════════════════════════════
# Aspect-Ratio-Preserving Resize
# ═════════════════════════════════════════════════════════════════════════════

def resize_panel(frame: np.ndarray, target_height: int,
                 target_width: int) -> np.ndarray:
    """
    Resize a feedback panel image to fit inside a (target_height × target_width)
    canvas while preserving aspect ratio.

    Behaviour:
      - Scale the image height to target_height.
      - If the scaled width is narrower than target_width, center it horizontally
        on a black canvas (letterbox / pillarbox).
      - If wider, center-crop to target_width.

    Args:
        frame:         BGR image (np.ndarray).
        target_height: Desired panel height in pixels.
        target_width:  Desired panel width in pixels.

    Returns:
        A BGR np.ndarray of exactly (target_height × target_width × 3).
    """
    if frame is None:
        return make_blank_frame(target_height, target_width)

    h, w = frame.shape[:2]
    if h == 0:
        return make_blank_frame(target_height, target_width)

    scale  = target_height / h
    new_w  = int(w * scale)
    resized = cv2.resize(frame, (new_w, target_height), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    if new_w >= target_width:
        x_start = (new_w - target_width) // 2
        canvas  = resized[:, x_start:x_start + target_width]
    else:
        x_offset = (target_width - new_w) // 2
        canvas[:, x_offset:x_offset + new_w] = resized

    return canvas


# ═════════════════════════════════════════════════════════════════════════════
# Convenience Loader — Auto-detect image vs. GIF
# ═════════════════════════════════════════════════════════════════════════════

def load_image_panel(filename: str, images_dir: str = None):
    """
    Auto-detect file type and load accordingly.

    Args:
        filename:   Name of the image or GIF file.
        images_dir: Optional override for the images directory path.

    Returns:
        For .gif  → ("gif",   list[np.ndarray]) — list of animation frames
        For other → ("image", np.ndarray | None) — single frame or None
    """
    if filename.lower().endswith(".gif"):
        frames = load_gif_frames(filename, images_dir)
        return ("gif", frames)
    else:
        img = load_image(filename, images_dir)
        return ("image", img)
