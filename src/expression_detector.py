"""
expression_detector.py — Real-Time Facial Expression & Gesture Detection
=========================================================================
Provides 11 independent expression scoring functions using MediaPipe's
FaceLandmarker blendshapes and HandLandmarker 3D landmarks.

Each function returns a float confidence score in the range [0.0, 1.0].
Thresholds are applied in main.py:
  - Hand-geometry expressions: 0.55
  - Blendshape-only expressions: 0.65

Supported MediaPipe blendshape keys (subset used here):
  jawOpen, mouthSmileLeft, mouthSmileRight, mouthPucker,
  eyeBlinkLeft, eyeBlinkRight, eyeSquintLeft, eyeSquintRight,
  browInnerUp, browDownLeft, browDownRight,
  noseSneerLeft, noseSneerRight, cheekSquintLeft, cheekSquintRight

Hand landmark indices (normalized 0–1, y increases downward):
  Wrist=0, Thumb tip=4, Index tip=8, Middle tip=12,
  Ring tip=16, Pinky tip=20
"""


# ── Internal Helpers ──────────────────────────────────────────────────────────

def _get_blendshape(blendshapes, name: str) -> float:
    """Return the score of a named blendshape, or 0.0 if not found."""
    for b in blendshapes:
        if b.category_name == name:
            return b.score
    return 0.0


def _clamp(value: float) -> float:
    """Clamp a value to [0.0, 1.0]."""
    return max(0.0, min(1.0, value))


# ═════════════════════════════════════════════════════════════════════════════
# 1. T-Pose / Timeout Gesture
#    Two hands forming a horizontal T-shape (arms outstretched symmetrically)
# ═════════════════════════════════════════════════════════════════════════════

def detect_t_pose(hands, blendshapes) -> float:
    """
    Detect a bilateral T-pose or timeout gesture.

    Criteria:
      - Exactly 2 hands detected
      - Both wrists at approximately the same vertical height (|Δy| < 0.2)
      - Wrists spread horizontally (|Δx| > 0.3 normalized units)

    Returns:
        Confidence score proportional to horizontal spread distance.
    """
    if len(hands) < 2:
        return 0.0

    wrist_0, wrist_1 = hands[0][0], hands[1][0]

    vertical_diff    = abs(wrist_0.y - wrist_1.y)
    horizontal_spread = abs(wrist_0.x - wrist_1.x)

    if vertical_diff > 0.2:
        return 0.0
    if horizontal_spread < 0.3:
        return 0.0

    return _clamp(horizontal_spread / 0.5)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Open-Mouth Bilateral Spread
#    Two hands spread wide + mouth open (jaw drop combined with arm spread)
# ═════════════════════════════════════════════════════════════════════════════

def detect_open_mouth_spread(hands, blendshapes) -> float:
    """
    Detect an open-mouth expression combined with bilateral hand spread.

    Criteria:
      - At least 2 hands
      - jawOpen blendshape > 0.5
      - Wrists spread horizontally > 0.3 normalized units

    Returns:
        Combined confidence based on jaw openness × spread.
    """
    if not blendshapes or len(hands) < 2:
        return 0.0

    jaw_open = _get_blendshape(blendshapes, "jawOpen")
    if jaw_open < 0.5:
        return 0.0

    wrist_0, wrist_1  = hands[0][0], hands[1][0]
    horizontal_spread = abs(wrist_0.x - wrist_1.x)
    if horizontal_spread < 0.3:
        return 0.0

    return _clamp(jaw_open * min(horizontal_spread / 0.4, 1.0))


# ═════════════════════════════════════════════════════════════════════════════
# 3. Serene Smile
#    Eyes gently closed + soft bilateral smile (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_serene_smile(blendshapes) -> float:
    """
    Detect a relaxed, eyes-closed smile expression.

    Criteria:
      - Both eye blink scores > 0.35 (eyes gently closed)
      - Both mouth smile scores > 0.25 (gentle smile)

    Returns:
        Combined confidence from average eye closure × average smile.
    """
    if not blendshapes:
        return 0.0

    blink_left  = _get_blendshape(blendshapes, "eyeBlinkLeft")
    blink_right = _get_blendshape(blendshapes, "eyeBlinkRight")
    smile_left  = _get_blendshape(blendshapes, "mouthSmileLeft")
    smile_right = _get_blendshape(blendshapes, "mouthSmileRight")

    if blink_left < 0.35 or blink_right < 0.35:
        return 0.0
    if smile_left < 0.25 or smile_right < 0.25:
        return 0.0

    avg_blink = (blink_left + blink_right) / 2.0
    avg_smile = (smile_left + smile_right) / 2.0

    return _clamp(avg_blink * avg_smile)


# ═════════════════════════════════════════════════════════════════════════════
# 4. Laughing Expression
#    Eyes tightly closed + strong bilateral smile (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_laughing(blendshapes) -> float:
    """
    Detect a laughing or intense joy expression.

    Criteria:
      - Both eye blink scores > 0.45 (eyes tightly closed)
      - Both mouth smile scores > 0.40 (strong smile)

    Returns:
        Combined confidence: min(blinks) × max(smiles).
    """
    if not blendshapes:
        return 0.0

    blink_left  = _get_blendshape(blendshapes, "eyeBlinkLeft")
    blink_right = _get_blendshape(blendshapes, "eyeBlinkRight")
    smile_left  = _get_blendshape(blendshapes, "mouthSmileLeft")
    smile_right = _get_blendshape(blendshapes, "mouthSmileRight")

    if blink_left < 0.45 or blink_right < 0.45:
        return 0.0
    if smile_left < 0.40 or smile_right < 0.40:
        return 0.0

    return _clamp(min(blink_left, blink_right) * max(smile_left, smile_right))


# ═════════════════════════════════════════════════════════════════════════════
# 5. Face-to-Hand Gesture
#    Single hand positioned in the face region (facepalm-like gesture)
# ═════════════════════════════════════════════════════════════════════════════

def detect_face_to_hand(hands, blendshapes) -> float:
    """
    Detect a hand-to-face gesture (e.g., facepalm, face touch).

    Criteria:
      - Exactly 1 hand detected
      - Wrist within the face region (0.25–0.75 x, 0.15–0.55 y)
      - jawOpen > 0.2 (slight mouth open — suggests reaction)

    Returns:
        Confidence proportional to jaw openness.
    """
    if not hands or not blendshapes:
        return 0.0
    if len(hands) != 1:
        return 0.0

    wrist    = hands[0][0]
    jaw_open = _get_blendshape(blendshapes, "jawOpen")

    if not (0.25 <= wrist.x <= 0.75):
        return 0.0
    if not (0.15 <= wrist.y <= 0.55):
        return 0.0
    if jaw_open < 0.2:
        return 0.0

    return _clamp(jaw_open * 2.0)


# ═════════════════════════════════════════════════════════════════════════════
# 6. Raised Hand Gesture
#    Single hand raised to head level, fingers pointing upward
# ═════════════════════════════════════════════════════════════════════════════

def detect_raised_hand(hands, blendshapes) -> float:
    """
    Detect a raised-hand or attention gesture.

    Criteria:
      - At least 1 hand with wrist above vertical midpoint (y < 0.45)
      - Index fingertip above the wrist (fingers pointing upward)

    Returns:
        Best confidence score across all detected hands.
    """
    if not hands:
        return 0.0

    best_score = 0.0
    for hand in hands:
        wrist     = hand[0]
        index_tip = hand[8]

        if wrist.y >= 0.45:
            continue
        if index_tip.y >= wrist.y:
            continue

        score      = _clamp((0.45 - wrist.y) / 0.3)
        best_score = max(best_score, score)

    return best_score


# ═════════════════════════════════════════════════════════════════════════════
# 7. Asymmetric Smile (Smirk)
#    Unequal activation of left vs. right mouth smile muscles (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_smirk(blendshapes) -> float:
    """
    Detect an asymmetric smile (smirk).

    Criteria:
      - Absolute difference between left/right smile > 0.15
      - Dominant side smile score > 0.25

    Returns:
        Confidence proportional to smile asymmetry.
    """
    if not blendshapes:
        return 0.0

    smile_left  = _get_blendshape(blendshapes, "mouthSmileLeft")
    smile_right = _get_blendshape(blendshapes, "mouthSmileRight")

    asymmetry = abs(smile_left - smile_right)
    if asymmetry < 0.15:
        return 0.0
    if max(smile_left, smile_right) < 0.25:
        return 0.0

    return _clamp(asymmetry * 2.0)


# ═════════════════════════════════════════════════════════════════════════════
# 8. Wink
#    Unilateral eye closure — one eye closed, other open (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_wink(blendshapes) -> float:
    """
    Detect a wink (one eye closed while the other remains open).

    Criteria:
      - Dominant eye blink > 0.40
      - Non-dominant eye blink < 0.15 (clear asymmetry)

    Returns:
        Confidence based on blink asymmetry score.
    """
    if not blendshapes:
        return 0.0

    blink_left  = _get_blendshape(blendshapes, "eyeBlinkLeft")
    blink_right = _get_blendshape(blendshapes, "eyeBlinkRight")

    max_blink = max(blink_left, blink_right)
    min_blink = min(blink_left, blink_right)

    if max_blink < 0.40:
        return 0.0
    if min_blink > 0.15:
        return 0.0

    return _clamp(max_blink - min_blink)


# ═════════════════════════════════════════════════════════════════════════════
# 9. Squint + Lip Pucker
#    Both eyes squinted + lips pursed (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_squint_pucker(blendshapes) -> float:
    """
    Detect a squinting + lip-pucker expression.

    Criteria:
      - Both eye squint scores > 0.35
      - mouthPucker > 0.30 (lips pushed forward)
      - jawOpen < 0.20 (mouth closed)

    Returns:
        Combined confidence: min(squints) × pucker.
    """
    if not blendshapes:
        return 0.0

    squint_left  = _get_blendshape(blendshapes, "eyeSquintLeft")
    squint_right = _get_blendshape(blendshapes, "eyeSquintRight")
    pucker       = _get_blendshape(blendshapes, "mouthPucker")
    jaw_open     = _get_blendshape(blendshapes, "jawOpen")

    if squint_left < 0.35 or squint_right < 0.35:
        return 0.0
    if pucker < 0.30:
        return 0.0
    if jaw_open > 0.20:
        return 0.0

    return _clamp(min(squint_left, squint_right) * pucker * 2.0)


# ═════════════════════════════════════════════════════════════════════════════
# 10. Jaw Drop / Surprised
#     Mouth wide open + raised inner brow (face only, no hands)
# ═════════════════════════════════════════════════════════════════════════════

def detect_jaw_drop(hands, blendshapes) -> float:
    """
    Detect a surprised jaw-drop expression.

    Criteria:
      - No hands detected (isolates pure facial expression)
      - jawOpen > 0.45 (wide open mouth)
      - browInnerUp > 0.25 (raised brows — surprise signal)

    Returns:
        Combined confidence: jaw_open × brow_raise.
    """
    if not blendshapes:
        return 0.0
    if len(hands) > 0:
        return 0.0

    jaw_open  = _get_blendshape(blendshapes, "jawOpen")
    brow_up   = _get_blendshape(blendshapes, "browInnerUp")

    if jaw_open < 0.45:
        return 0.0
    if brow_up < 0.25:
        return 0.0

    return _clamp(jaw_open * brow_up * 4.0)


# ═════════════════════════════════════════════════════════════════════════════
# 11. Narrowed-Eye Sneer
#     Minimal squint + slight nasal sneer + closed mouth (face only)
# ═════════════════════════════════════════════════════════════════════════════

def detect_narrowed_sneer(blendshapes) -> float:
    """
    Detect a suspicious or skeptical expression with a narrowed-eye sneer.

    Criteria:
      - Both eye squints < 0.15 (not squinting too hard — narrow but alert)
      - At least one noseSneer > 0.10 (slight upper lip curl)
      - jawOpen < 0.15 (mouth closed — composed expression)

    Returns:
        Confidence based on sneer intensity and eye openness.
    """
    if not blendshapes:
        return 0.0

    squint_left  = _get_blendshape(blendshapes, "eyeSquintLeft")
    squint_right = _get_blendshape(blendshapes, "eyeSquintRight")
    sneer_left   = _get_blendshape(blendshapes, "noseSneerLeft")
    sneer_right  = _get_blendshape(blendshapes, "noseSneerRight")
    jaw_open     = _get_blendshape(blendshapes, "jawOpen")

    if squint_left > 0.15 or squint_right > 0.15:
        return 0.0
    if sneer_left < 0.10 and sneer_right < 0.10:
        return 0.0
    if jaw_open > 0.15:
        return 0.0

    return _clamp(
        (sneer_left + sneer_right) * (1.0 - squint_left) * (1.0 - squint_right)
    )


# ═════════════════════════════════════════════════════════════════════════════
# 12. Idle
#     Fallback state — no significant expression detected
# ═════════════════════════════════════════════════════════════════════════════

def detect_idle() -> float:
    """
    Idle / neutral state. Always returns 0.0.
    Used as the fallback when no other expression exceeds its threshold.
    """
    return 0.0
