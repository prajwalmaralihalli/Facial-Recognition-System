"""
debug.py — Real-Time Debug Overlay
-----------------------------------
Provides a toggleable debug visualization overlay for the Facial
Recognition System. Renders face mesh landmarks, hand skeletons,
and prints live MediaPipe blendshape values to the terminal.

Toggle on/off at runtime with the 'D' key.

Usage in main.py:
    from debug import DebugOverlay

    overlay = DebugOverlay()

    # Inside main loop:
    overlay.handle_key(key)                               # key from cv2.waitKey
    overlay.draw(frame, face_landmarks, hand_list, blendshapes)
"""

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Hand Skeleton Connections (MediaPipe 21-landmark topology)
# ─────────────────────────────────────────────────────────────────────────────

HAND_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index finger
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle finger
    (5, 9), (9, 10), (10, 11), (11, 12),
    # Ring finger
    (9, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (13, 17), (17, 18), (18, 19), (19, 20),
    # Palm base
    (0, 17),
]

# Color cycle for up to 4 simultaneously detected hands
HAND_COLORS = [
    (0, 255, 255),   # cyan   — hand 0
    (255, 0, 255),   # magenta — hand 1
    (255, 165, 0),   # orange — hand 2
    (0, 128, 255),   # blue   — hand 3
]


class DebugOverlay:
    """
    Toggleable real-time debug overlay for the Facial Recognition System.

    When enabled, draws:
      - Face mesh: 468 green landmark dots with key indices labeled
      - Hand skeleton: Joint circles + bone connections for each hand
      - Terminal output: Active blendshape values above a threshold

    Attributes:
        enabled (bool):           Whether the overlay is currently active.
        print_threshold (float):  Minimum blendshape score to include in output.
    """

    def __init__(self, enabled: bool = False, print_threshold: float = 0.1):
        self.enabled         = enabled
        self.print_threshold = print_threshold
        self._frame_count    = 0
        self._print_interval = 15  # print blendshapes every N frames

    # ──────────────────────────────────────────────────────────────────────────
    # Toggle
    # ──────────────────────────────────────────────────────────────────────────

    def handle_key(self, key: int) -> None:
        """
        Handle keyboard input. Toggles debug mode when 'D' is pressed.

        Args:
            key: Return value of cv2.waitKey() & 0xFF.
        """
        if key == ord("d"):
            self.enabled = not self.enabled
            state = "ON" if self.enabled else "OFF"
            print(f"[debug] Overlay toggled {state}")

    # ──────────────────────────────────────────────────────────────────────────
    # Main Draw Entry Point
    # ──────────────────────────────────────────────────────────────────────────

    def draw(self, frame, face_landmarks=None, hand_landmarks_list=None,
             blendshapes=None) -> None:
        """
        Draw all debug overlays onto `frame` in-place.

        Args:
            frame:               BGR webcam frame (np.ndarray).
            face_landmarks:      Single face landmark list (468 points) or None.
            hand_landmarks_list: List of hand landmark lists (each 21 pts) or None.
            blendshapes:         Single face blendshapes list or None.
        """
        if not self.enabled:
            return

        self._frame_count += 1

        if face_landmarks:
            self._draw_face_mesh(frame, face_landmarks)

        if hand_landmarks_list:
            self._draw_hand_skeletons(frame, hand_landmarks_list)

        if blendshapes and self._frame_count % self._print_interval == 0:
            self._print_blendshapes(blendshapes)

        self._draw_status_badge(frame)

    # ──────────────────────────────────────────────────────────────────────────
    # Face Mesh (468 landmarks)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_face_mesh(self, frame, face_landmarks) -> None:
        """Draw all 468 face landmarks as small green dots with key indices labeled."""
        h, w = frame.shape[:2]
        for lm in face_landmarks:
            cx = int(lm.x * w)
            cy = int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        # Label anatomically significant landmarks
        key_landmarks = {
            1:   "nose",
            13:  "upper_lip",
            14:  "lower_lip",
            61:  "L_mouth",
            291: "R_mouth",
            152: "chin",
            234: "L_ear",
            454: "R_ear",
        }
        for idx, label in key_landmarks.items():
            if idx < len(face_landmarks):
                lm = face_landmarks[idx]
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 3, (0, 200, 255), -1)
                cv2.putText(frame, f"{idx}", (cx + 4, cy - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 200, 255), 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Hand Skeletons (21 landmarks per hand, up to 2 hands)
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_hand_skeletons(self, frame, hand_landmarks_list) -> None:
        """Draw skeleton connections and joint markers for each detected hand."""
        h, w = frame.shape[:2]

        for idx, hand in enumerate(hand_landmarks_list):
            color = HAND_COLORS[idx % len(HAND_COLORS)]
            pts   = [(int(lm.x * w), int(lm.y * h)) for lm in hand]

            # Draw bones (connections between landmark pairs)
            for start, end in HAND_CONNECTIONS:
                cv2.line(frame, pts[start], pts[end], color, 2)

            # Draw joint markers (larger circles for fingertips)
            for i, pt in enumerate(pts):
                radius = 5 if i in (4, 8, 12, 16, 20) else 3
                cv2.circle(frame, pt, radius, color, -1)
                cv2.putText(frame, str(i), (pt[0] + 4, pt[1] - 4),
                            cv2.FONT_HERSHEY_PLAIN, 0.6, color, 1)

    # ──────────────────────────────────────────────────────────────────────────
    # Blendshape Terminal Output
    # ──────────────────────────────────────────────────────────────────────────

    def _print_blendshapes(self, blendshapes) -> None:
        """Print active blendshapes (above threshold) to the terminal."""
        active = [
            (bs.category_name, bs.score)
            for bs in blendshapes
            if bs.score > self.print_threshold
        ]

        if not active:
            return

        active.sort(key=lambda x: x[1], reverse=True)

        print("-" * 55)
        print(f"[debug] Active blendshapes (> {self.print_threshold}):")
        for name, score in active:
            bar = "#" * int(score * 20)
            print(f"  {name:<30s} {score:.3f}  {bar}")

    # ──────────────────────────────────────────────────────────────────────────
    # Status Badge
    # ──────────────────────────────────────────────────────────────────────────

    def _draw_status_badge(self, frame) -> None:
        """Render a 'DEBUG ON' badge in the top-right corner of the frame."""
        text  = "DEBUG ON"
        font  = cv2.FONT_HERSHEY_SIMPLEX
        scale, thickness = 0.5, 1
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        h, w = frame.shape[:2]
        x = w - tw - 15
        y = 25
        cv2.rectangle(frame, (x - 5, y - th - 5), (x + tw + 5, y + 5),
                      (0, 0, 200), -1)
        cv2.putText(frame, text, (x, y), font, scale,
                    (255, 255, 255), thickness, cv2.LINE_AA)
