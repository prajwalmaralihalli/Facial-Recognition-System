"""
Facial Recognition System — main.py
=====================================
Real-time facial expression and hand gesture recognition using
the MediaPipe Tasks API with OpenCV.

Architecture:
  Webcam frame → MediaPipe (Face Landmarker + Hand Landmarker)
               → Expression scoring (expression_detector.py)
               → Gesture stabilization buffer
               → Side-by-side display (webcam | visual feedback)

Module overview:
  expression_detector.py  — detect_* functions (11 expression categories)
  media_loader.py         — load_image, get_current_frame, resize_panel
  debug.py                — DebugOverlay (toggle with 'D')
  download_models.py      — Automatic MediaPipe model downloader

Controls:  Q = quit,  D = toggle debug overlay
"""

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import sys
from pathlib import Path

# ── Project module imports ────────────────────────────────────────────────────
from download_models import download_all, MODELS_DIR, MODELS
from debug import DebugOverlay
from media_loader import load_image_panel, get_current_frame, resize_panel, make_blank_frame
from expression_detector import (
    detect_t_pose, detect_open_mouth_spread, detect_serene_smile,
    detect_laughing, detect_face_to_hand, detect_raised_hand,
    detect_smirk, detect_wink, detect_squint_pucker,
    detect_jaw_drop, detect_narrowed_sneer, detect_idle,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

FACE_MODEL_PATH = MODELS_DIR / "face_landmarker.task"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

# ── Gesture stabilization (frame-based debouncing) ──
CONFIRM_FRAMES  = 6    # consecutive frames required to trigger an expression
HOLD_FRAMES     = 40   # frames to lock the current display panel
COOLDOWN_FRAMES = 15   # frames to ignore new detections after returning to idle

IMAGES_DIR = os.path.join(os.path.dirname(__file__), "..", "images")

# ── Expression → visual feedback image mapping ──
EXPRESSION_IMAGE_MAP = {
    "t_pose":            "shaq.jpeg",
    "open_mouth_spread": "real madrid.jpeg",
    "serene_smile":      "black.jpeg",
    "laughing":          "girl.jpeg",
    "face_to_hand":      "j hill.jpeg",
    "raised_hand":       "pique.jpeg",
    "smirk":             "speed.jpeg",
    "wink":              "doakes.jpeg",
    "squint_pucker":     "speed.jpeg",
    "jaw_drop":          "patrick.jpeg",
    "narrowed_sneer":    "doakes.jpeg",
    "idle":              "understandable.jpeg",
}

# ── Confidence thresholds by detection method ──
HAND_EXPRESSIONS = {"t_pose", "open_mouth_spread", "face_to_hand", "raised_hand"}
HAND_THRESHOLD = 0.55   # lower threshold — hand geometry is explicit
FACE_THRESHOLD = 0.65   # higher threshold — blendshape scores need stronger signal


# ─────────────────────────────────────────────────────────────────────────────
# MediaPipe Detector Creation (Tasks API — IMAGE mode)
# ─────────────────────────────────────────────────────────────────────────────

def create_face_landmarker():
    """Initialize the Face Landmarker with blendshape output enabled."""
    opts = vision.FaceLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(FACE_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=False,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(opts)


def create_hand_landmarker():
    """Initialize the Hand Landmarker for up to 2 hands."""
    opts = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(HAND_MODEL_PATH)),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3,
    )
    return vision.HandLandmarker.create_from_options(opts)


# ─────────────────────────────────────────────────────────────────────────────
# Visual Feedback Panel State
# ─────────────────────────────────────────────────────────────────────────────

_panel_type     = None        # "gif" | "image" | None
_panel_data     = None        # list[ndarray] (gif) | ndarray (image) | None
_gif_index      = 0
_target_h       = 480
_target_w       = 640
_idle_frame     = None        # set after webcam opens
_current_loaded = None        # expression name of currently loaded panel


def _set_dimensions(h: int, w: int):
    global _target_h, _target_w, _idle_frame
    _target_h, _target_w = h, w
    _idle_frame = make_blank_frame(h, w, "Waiting for expression...")


def _load_expression_panel(expression: str) -> bool:
    """Load the visual feedback panel for a given expression. Returns True on success."""
    global _panel_type, _panel_data, _gif_index, _current_loaded

    if expression == _current_loaded:
        return True

    _current_loaded = expression
    _panel_type = None
    _panel_data = None
    _gif_index  = 0

    if expression is None or expression not in EXPRESSION_IMAGE_MAP:
        return False

    filename = EXPRESSION_IMAGE_MAP[expression]
    ptype, pdata = load_image_panel(filename, IMAGES_DIR)

    if ptype == "gif":
        if not pdata:
            return False
        _panel_type = "gif"
        _panel_data = pdata
    elif ptype == "image":
        if pdata is None:
            return False
        _panel_type = "image"
        _panel_data = pdata

    return True


def _get_panel_frame() -> np.ndarray:
    """Return the current visual feedback panel frame, sized to target dimensions."""
    global _gif_index

    if _panel_type == "gif" and _panel_data:
        frame, _gif_index = get_current_frame(_panel_data, _gif_index)
        if frame is not None:
            return resize_panel(frame, _target_h, _target_w)

    elif _panel_type == "image" and _panel_data is not None:
        return resize_panel(_panel_data, _target_h, _target_w)

    return _idle_frame.copy() if _idle_frame is not None else make_blank_frame()


# ─────────────────────────────────────────────────────────────────────────────
# Visualization Helper
# ─────────────────────────────────────────────────────────────────────────────

def draw_detection_labels(frame, detections: list) -> None:
    """Render expression labels with confidence percentages on the webcam frame."""
    y = 40
    font = cv2.FONT_HERSHEY_SIMPLEX
    for text, color in detections:
        (tw, th), _ = cv2.getTextSize(text, font, 0.8, 2)
        cv2.rectangle(frame, (10, y - th - 5), (20 + tw, y + 5), (0, 0, 0), -1)
        cv2.putText(frame, text, (15, y), font, 0.8, color, 2, cv2.LINE_AA)
        y += 45


# ─────────────────────────────────────────────────────────────────────────────
# Startup Validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_images_dir() -> bool:
    """Verify the images directory exists and contains required feedback images."""
    p = Path(IMAGES_DIR)
    if not p.exists():
        print(f"\n[ERROR] Images directory not found: {p.resolve()}")
        print("        Create it and add expression feedback image files:")
        print("          mkdir images")
        return False
    files = [f for f in p.iterdir() if f.is_file()]
    if not files:
        print(f"\n[ERROR] Images directory is empty: {p.resolve()}")
        print("        Add image/GIF files to ./images/ before running.")
        return False

    found, missing = 0, []
    seen = set()
    for expression, filename in EXPRESSION_IMAGE_MAP.items():
        if filename in seen:
            continue
        seen.add(filename)
        if (p / filename).exists():
            found += 1
        else:
            missing.append(filename)

    print(f"  Feedback images: {found} found, {len(missing)} missing")
    if missing:
        for f in missing:
            print(f"    [MISS] {f}")
    return True


# ─────────────────────────────────────────────────────────────────────────────
# Expression Evaluation — Score all 11 categories every frame
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_all_expressions(hands, blendshapes):
    """
    Score every expression category. Returns dict of {expression_name: score}.
    All categories are evaluated every frame — no early-exit short-circuiting.
    """
    scores = {}

    # Priority 1: T-Pose / Timeout (two hands, horizontal T-shape)
    scores["t_pose"]            = detect_t_pose(hands, blendshapes)

    # Priority 2: Open-mouth bilateral spread (two hands + jaw open)
    scores["open_mouth_spread"] = detect_open_mouth_spread(hands, blendshapes)

    # Priority 3: Serene smile (eyes closed + gentle smile — face only)
    scores["serene_smile"]      = detect_serene_smile(blendshapes)

    # Priority 4: Laughing (eyes closed + strong smile — face only)
    scores["laughing"]          = detect_laughing(blendshapes)

    # Priority 5: Face-to-hand gesture (single hand in face region)
    scores["face_to_hand"]      = detect_face_to_hand(hands, blendshapes)

    # Priority 6: Raised hand (single hand above mid-screen)
    scores["raised_hand"]       = detect_raised_hand(hands, blendshapes)

    # Priority 7: Asymmetric smile / smirk (face only)
    scores["smirk"]             = detect_smirk(blendshapes)

    # Priority 8: Wink — unilateral eye closure (face only)
    scores["wink"]              = detect_wink(blendshapes)

    # Priority 9: Squint + lip pucker (face only)
    scores["squint_pucker"]     = detect_squint_pucker(blendshapes)

    # Priority 10: Jaw drop / surprised (face only, no hands)
    scores["jaw_drop"]          = detect_jaw_drop(hands, blendshapes)

    # Priority 11: Narrowed-eye sneer (face only)
    scores["narrowed_sneer"]    = detect_narrowed_sneer(blendshapes)

    # Idle fallback (always 0.0 — used when nothing else triggers)
    scores["idle"]              = detect_idle()

    return scores


def pick_winner(scores):
    """
    Select the highest-confidence expression above its threshold.

    Returns:
        (expression_name, score): Best match, or ('idle', 0.0) if nothing qualifies.
    """
    best_name  = "idle"
    best_score = 0.0

    for name, score in scores.items():
        if name == "idle":
            continue
        threshold = HAND_THRESHOLD if name in HAND_EXPRESSIONS else FACE_THRESHOLD
        if score >= threshold and score > best_score:
            best_name  = name
            best_score = score

    return best_name, best_score


# ─────────────────────────────────────────────────────────────────────────────
# Main Application
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("  Facial Recognition System")
    print("  Real-Time Expression & Gesture Detection")
    print("=" * 60)

    # ── 1. Download MediaPipe models (auto-skips if already present) ──
    try:
        download_all()
    except Exception as exc:
        print(f"\n[ERROR] Model download failed: {exc}")
        print("        Check your internet connection, or place .task files")
        print("        manually in ./models/")
        sys.exit(1)

    for name in MODELS:
        if not (MODELS_DIR / name).exists():
            print(f"\n[ERROR] Model file missing: {name}")
            print("        Run 'python src/download_models.py' or place it in ./models/")
            sys.exit(1)

    # ── 2. Initialize MediaPipe detectors ──
    try:
        face_detector = create_face_landmarker()
        hand_detector = create_hand_landmarker()
    except Exception as exc:
        print(f"\n[ERROR] Failed to initialize MediaPipe detectors: {exc}")
        sys.exit(1)

    # ── 3. Validate images directory ──
    if not validate_images_dir():
        sys.exit(1)

    # ── 4. Initialize debug overlay (toggle with 'D') ──
    debug_overlay = DebugOverlay(enabled=False)

    # ── 5. Open webcam ──
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("\n[ERROR] Cannot open webcam (index 0).")
        print("        Try changing VideoCapture(0) to VideoCapture(1).")
        sys.exit(1)

    ret, test_frame = cap.read()
    if not ret:
        print("[ERROR] Webcam opened but could not read a frame.")
        cap.release()
        sys.exit(1)

    h, w = test_frame.shape[:2]
    _set_dimensions(h, w)
    print(f"  Webcam resolution: {w}x{h}")

    # ── Stabilization state ──
    current_expression = "idle"
    pending_expression = None
    pending_count      = 0
    hold_remaining     = 0
    cooldown_remaining = 0
    frame_count        = 0

    # Pre-load idle panel so the right panel is populated immediately
    _load_expression_panel("idle")
    print("  System ready.")
    print("  Controls: Q = quit  |  D = toggle debug overlay\n")

    # ── 6. Main inference loop ──
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — exiting.")
            break

        frame = cv2.flip(frame, 1)  # mirror for natural interaction
        frame_count += 1

        # ── MediaPipe Inference ──
        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        face_result = face_detector.detect(mp_img)
        hand_result = hand_detector.detect(mp_img)

        # ── Extract landmark results ──
        face_landmarks = face_result.face_landmarks[0] if face_result.face_landmarks else None
        blendshapes    = face_result.face_blendshapes[0] if face_result.face_blendshapes else []
        hands          = hand_result.hand_landmarks if hand_result.hand_landmarks else []

        # ── Score all 11 expression categories ──
        scores = evaluate_all_expressions(hands, blendshapes)
        active_expression, active_score = pick_winner(scores)

        # ── Periodic debug logging ──
        if frame_count % 30 == 0:
            print(f"\n[Frame {frame_count}] hands={len(hands)}")
            for i, h_lm in enumerate(hands):
                wr = h_lm[0]
                print(f"  hand[{i}] wrist x={wr.x:.3f} y={wr.y:.3f}")
            if not hands:
                print("  (no hands detected)")
            sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for name, score in sorted_scores:
                thresh = HAND_THRESHOLD if name in HAND_EXPRESSIONS else FACE_THRESHOLD
                marker = " <<<" if score >= thresh else ""
                print(f"  {name:<20s} {score:.3f}  (threshold={thresh:.2f}){marker}")

        # ── Build active detection labels for overlay ──
        detections = []
        for name, sc in sorted(scores.items(), key=lambda x: x[1], reverse=True):
            if name == "idle":
                continue
            thresh = HAND_THRESHOLD if name in HAND_EXPRESSIONS else FACE_THRESHOLD
            if sc >= thresh:
                label     = name.upper().replace("_", " ")
                conf_text = f"{label}  {int(sc * 100)}%"
                detections.append((conf_text, (0, 255, 0)))

        # ── Winner label in top-left corner ──
        if active_expression != "idle":
            overlay_text = f"{active_expression.replace('_', ' ')}  {int(active_score * 100)}%"
            font = cv2.FONT_HERSHEY_SIMPLEX
            (tw, th), _ = cv2.getTextSize(overlay_text, font, 1.0, 2)
            cv2.rectangle(frame, (8, 8), (18 + tw, 18 + th), (0, 0, 0), -1)
            cv2.putText(frame, overlay_text, (12, 12 + th), font, 1.0,
                        (0, 255, 100), 2, cv2.LINE_AA)

        # ── Console notification on new expression ──
        if active_expression != "idle" and active_expression != current_expression:
            print(f"[EXPRESSION DETECTED] {active_expression}  (confidence={active_score:.2f})")

        # ── Draw detection labels ──
        if detections:
            draw_detection_labels(frame, detections)

        # ── Debug overlay (face mesh + hand skeleton + blendshapes) ──
        debug_overlay.draw(frame, face_landmarks, hands, blendshapes)

        # ── Gesture stabilization pipeline ──
        if hold_remaining > 0:
            hold_remaining -= 1
        if cooldown_remaining > 0:
            cooldown_remaining -= 1

        if hold_remaining > 0:
            pending_expression = None
            pending_count = 0
        elif cooldown_remaining > 0 and current_expression == "idle" and active_expression != "idle":
            pending_expression = None
            pending_count = 0
        else:
            if active_expression == pending_expression:
                pending_count += 1
            else:
                pending_expression = active_expression
                pending_count = 1

            if pending_count >= CONFIRM_FRAMES and pending_expression != current_expression:
                old_expression = current_expression
                _load_expression_panel(pending_expression)
                current_expression = pending_expression
                print(f"[PANEL SWITCH] → {current_expression}")
                pending_expression = None
                pending_count = 0
                hold_remaining = HOLD_FRAMES
                if current_expression == "idle" and old_expression != "idle":
                    cooldown_remaining = COOLDOWN_FRAMES

        # ── Compose side-by-side display ──
        panel_frame = _get_panel_frame()
        cv2.putText(frame, "Q=quit  D=debug", (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1, cv2.LINE_AA)

        combined = np.hstack((frame, panel_frame))
        cv2.imshow("Facial Recognition System", combined)

        # ── Key handling ──
        key = cv2.waitKey(1) & 0xFF
        debug_overlay.handle_key(key)
        if key == ord("q"):
            break

    # ── Cleanup ──
    cap.release()
    cv2.destroyAllWindows()
    print("\nSession ended.")


if __name__ == "__main__":
    main()
