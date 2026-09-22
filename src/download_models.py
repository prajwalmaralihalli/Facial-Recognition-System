"""
download_models.py — MediaPipe Model Downloader
-------------------------------------------------
Downloads the FaceLandmarker and HandLandmarker .task files from
Google Storage into the ./models/ directory on first run.

Models are stored in float16 format for optimized TFLite inference
on both CPU and GPU backends.

Usage:
    python src/download_models.py

Or imported and called automatically by main.py at startup:
    from download_models import download_all
    download_all()
"""

import urllib.request
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Model Registry
# ─────────────────────────────────────────────────────────────────────────────

MODELS = {
    "face_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    ),
    "hand_landmarker.task": (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    ),
}

MODELS_DIR = Path(__file__).parent.parent / "models"


# ─────────────────────────────────────────────────────────────────────────────
# Download Progress Hook
# ─────────────────────────────────────────────────────────────────────────────

def _make_progress_hook(filename: str):
    """Return a urlretrieve reporthook that renders a live ASCII progress bar."""
    BAR_WIDTH = 40

    def hook(block_num: int, block_size: int, total_size: int) -> None:
        downloaded = block_num * block_size

        if total_size > 0:
            pct    = min(downloaded / total_size, 1.0)
            filled = int(BAR_WIDTH * pct)
            bar    = "#" * filled + "-" * (BAR_WIDTH - filled)
            dl_mb  = downloaded / 1_048_576
            tot_mb = total_size / 1_048_576
            sys.stdout.write(
                f"\r  [{bar}] {pct:5.1%}  {dl_mb:.1f} / {tot_mb:.1f} MB"
            )
        else:
            dl_mb = downloaded / 1_048_576
            sys.stdout.write(f"\r  Downloaded {dl_mb:.1f} MB...")

        sys.stdout.flush()

        if total_size > 0 and downloaded >= total_size:
            sys.stdout.write("\n")

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Individual Model Downloader
# ─────────────────────────────────────────────────────────────────────────────

def download_model(filename: str, url: str) -> None:
    """
    Download a single model file if it is not already present.

    Skips the download if the file exists (size is checked implicitly by
    existence). Cleans up any partial file if the download fails.

    Args:
        filename: Local filename to save (relative to MODELS_DIR).
        url:      Source URL for the model file.
    """
    dest = MODELS_DIR / filename

    if dest.exists():
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  [SKIP] {filename} already present ({size_mb:.1f} MB)")
        return

    print(f"  [DOWN] {filename}")
    print(f"         {url}")

    try:
        urllib.request.urlretrieve(url, dest, reporthook=_make_progress_hook(filename))
        size_mb = dest.stat().st_size / 1_048_576
        print(f"  [ OK ] Saved → {dest}  ({size_mb:.1f} MB)")
    except Exception as exc:
        if dest.exists():
            dest.unlink()   # clean up partial download
        print(f"\n  [FAIL] Could not download {filename}: {exc}")
        print("         Try placing the file manually in ./models/")


# ─────────────────────────────────────────────────────────────────────────────
# Download All Models
# ─────────────────────────────────────────────────────────────────────────────

def download_all() -> None:
    """
    Download all required MediaPipe model files.

    Called automatically by main.py at startup. Models that already exist
    are skipped without re-downloading.
    """
    print()
    print("=" * 60)
    print("  MediaPipe Model Downloader")
    print("=" * 60)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    print(f"  Models directory: {MODELS_DIR.resolve()}")
    print()

    for filename, url in MODELS.items():
        download_model(filename, url)
        print()

    # Final status report
    print("-" * 60)
    all_ok = True
    for filename in MODELS:
        dest = MODELS_DIR / filename
        if dest.exists():
            size_mb = dest.stat().st_size / 1_048_576
            print(f"  ✓  {filename}  ({size_mb:.1f} MB)")
        else:
            print(f"  ✗  {filename}  MISSING")
            all_ok = False

    print()
    if all_ok:
        print("  All models ready. Run:  python src/main.py")
    else:
        print("  One or more models are missing. Check the errors above.")
    print("=" * 60)
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Entry Point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    download_all()
