# 🎭 Facial Recognition System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/MediaPipe-Tasks%20API-orange?logo=google&logoColor=white" alt="MediaPipe"/>
  <img src="https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv&logoColor=white" alt="OpenCV"/>
  <img src="https://img.shields.io/badge/NumPy-1.24%2B-yellow?logo=numpy&logoColor=white" alt="NumPy"/>
  <img src="https://img.shields.io/badge/Real--Time-30fps-brightgreen" alt="Real-Time"/>
  <img src="https://img.shields.io/badge/Expressions-11%20Categories-purple" alt="Expressions"/>
</p>

<p align="center">
  A real-time facial expression and hand gesture recognition system built with <strong>Google MediaPipe Tasks API</strong> and <strong>OpenCV</strong>. Detects 11 distinct facial expression and gesture categories from a live webcam feed with frame-by-frame confidence scoring.
</p>

---

## 📌 Table of Contents

- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Detected Expressions & Gestures](#-detected-expressions--gestures)
- [Getting Started](#-getting-started)
- [Usage & Controls](#-usage--controls)
- [Project Structure](#-project-structure)
- [How It Works](#-how-it-works)

---

## ✨ Features

- 🔍 **Real-Time Detection** — Processes live webcam frames at up to 30 FPS
- 🧠 **Dual-Model Inference** — Simultaneously runs a **Face Landmarker** (468 landmarks + 52 blendshapes) and a **Hand Landmarker** (21 landmarks × 2 hands)
- 📊 **Confidence Scoring** — Every expression is assigned a continuous float confidence score (0.0 – 1.0); the best candidate above threshold wins
- 🎯 **11 Expression Categories** — Covers both facial blendshape-based and hand-landmark-based detections
- 🔄 **Gesture Stabilization** — Frame-buffer debouncing prevents flickering (6-frame confirmation window, 40-frame hold, 15-frame cooldown)
- 🐛 **Debug Mode** — Toggle face mesh (468 dots), hand skeleton (21 joints), and live blendshape terminal output with `D`
- 🖥️ **Side-by-Side Display** — Webcam feed on the left; expression-matched visual feedback panel on the right

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        WEBCAM INPUT (BGR)                        │
└──────────────────────────────┬──────────────────────────────────┘
                               │ cv2.flip + RGB conversion
                    ┌──────────▼──────────┐
                    │  MediaPipe MP Image  │
                    └──────────┬──────────┘
              ┌────────────────┴────────────────┐
              ▼                                 ▼
  ┌───────────────────────┐         ┌───────────────────────┐
  │   Face Landmarker     │         │   Hand Landmarker     │
  │  (float16 .task model)│         │  (float16 .task model)│
  │  → 468 face landmarks │         │  → 21 landmarks×2 hand│
  │  → 52 blendshapes     │         │  → handedness label   │
  └───────────┬───────────┘         └───────────┬───────────┘
              └────────────────┬────────────────┘
                               ▼
              ┌────────────────────────────────┐
              │     Expression Detector        │
              │  expression_detector.py        │
              │  Score all 11 categories       │
              │  Pick winner above threshold   │
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │    Gesture Stabilizer          │
              │  6-frame confirm buffer        │
              │  40-frame hold lock            │
              │  15-frame cooldown             │
              └────────────────┬───────────────┘
                               ▼
              ┌────────────────────────────────┐
              │     Side-by-Side Display       │
              │  Left:  Webcam + overlays      │
              │  Right: Visual feedback panel  │
              └────────────────────────────────┘
```

---

## 🛠 Tech Stack

| Component | Library / Tool | Purpose |
|---|---|---|
| Face Detection | MediaPipe Face Landmarker | 468 landmarks + 52 AU blendshapes |
| Hand Detection | MediaPipe Hand Landmarker | 21 landmarks, up to 2 hands |
| Computer Vision | OpenCV 4.x | Frame capture, display, drawing |
| Numerical Computing | NumPy | Frame compositing, array ops |
| Model Format | MediaPipe Tasks API (float16) | Optimized TFLite inference |
| Language | Python 3.9+ | Core application |

---

## 🎭 Detected Expressions & Gestures

| # | Category | Detection Method | Key Signals |
|---|---|---|---|
| 1 | **T-Pose / Timeout** | Hand landmarks | Two wrists at equal height, spread > 0.3 normalized units |
| 2 | **Open-Mouth Spread** | Face + Hand | `jawOpen` > 0.5, two hands spread bilaterally |
| 3 | **Serene Smile** | Face blendshapes | Both eyes closed (blink > 0.35), gentle bilateral smile |
| 4 | **Laughing** | Face blendshapes | Both eyes closed (blink > 0.45), strong smile (> 0.4) |
| 5 | **Face-to-Hand Gesture** | Face + Hand | Single wrist in face region (0.25–0.75 x, 0.15–0.55 y), jaw open |
| 6 | **Raised Hand / Shush** | Hand landmarks | Wrist above mid-screen (y < 0.45), index tip above wrist |
| 7 | **Asymmetric Smile (Smirk)** | Face blendshapes | Left/right smile difference > 0.15 |
| 8 | **Wink** | Face blendshapes | One eye blink > 0.4, other < 0.15 |
| 9 | **Squint + Lip Pucker** | Face blendshapes | Both eye squints > 0.35, `mouthPucker` > 0.3 |
| 10 | **Jaw Drop / Surprised** | Face blendshapes | `jawOpen` > 0.45, `browInnerUp` > 0.25, no hands |
| 11 | **Narrowed-Eye Sneer** | Face blendshapes | Low squint, `noseSneer` > 0.1, jaw closed |

**Thresholds:**
- Hand-gesture categories: **0.55** confidence
- Face-only categories: **0.65** confidence

---

## 🚀 Getting Started

### Prerequisites

- Python **3.9 or higher**
- A working **webcam**
- Internet connection (first run downloads ~20 MB of MediaPipe model files)

### 1. Clone the Repository

```bash
git clone https://github.com/prajwalmaralihalli/Facial-Recognition-System.git
cd Facial-Recognition-System
```

### 2. Automated Setup (Windows — PowerShell)

```powershell
.\setup.ps1
```

This script will:
- Create `images/` and `models/` directories
- Create a Python virtual environment (`venv/`)
- Install all dependencies from `requirements.txt`

### 3. Manual Setup (Cross-Platform)

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Add Expression Feedback Images

Place your visual feedback images in the `images/` folder. These are displayed on the right panel when an expression is detected.

The expected filenames are configured in `src/main.py` under `EXPRESSION_IMAGE_MAP`.

### 5. Run the Application

```bash
# Activate venv first, then:
python src/main.py
```

On first run, the application will **automatically download** the MediaPipe model files (~20 MB total) into `models/`.

---

## 🎮 Usage & Controls

| Key | Action |
|---|---|
| `Q` | Quit the application |
| `D` | Toggle debug overlay (face mesh, hand skeleton, blendshape values) |

### Debug Overlay

When debug mode is **ON**:
- 🟢 **Green dots** — All 468 face mesh landmarks
- 🟡 **Cyan/Magenta lines** — Hand skeleton for each detected hand (up to 2)
- 🖥️ **Terminal** — Live blendshape values above 0.1 threshold, printed every 15 frames

---

## 📁 Project Structure

```
Facial-Recognition-System/
├── README.md
├── requirements.txt
├── setup.ps1                    # Automated Windows setup
├── images/                      # Expression feedback images (not tracked)
├── models/                      # MediaPipe .task model files (not tracked)
└── src/
    ├── main.py                  # Application entry point & main loop
    ├── expression_detector.py   # All 11 expression scoring functions
    ├── media_loader.py          # Image/GIF loading and frame utilities
    ├── debug.py                 # DebugOverlay class
    └── download_models.py       # Automatic MediaPipe model downloader
```

---

## 🔬 How It Works

### 1. Dual-Model MediaPipe Inference

Each video frame is converted to RGB and wrapped in a `mediapipe.Image` object. Two landmarker models run in **IMAGE mode** (synchronous, no timestamps needed):

```python
face_result = face_landmarker.detect(mp_image)   # 468 landmarks + 52 blendshapes
hand_result = hand_landmarker.detect(mp_image)   # 21 landmarks × up to 2 hands
```

### 2. Blendshape-Based Expression Scoring

MediaPipe's `FaceLandmarker` outputs 52 **Action Unit blendshapes** (e.g., `jawOpen`, `mouthSmileLeft`, `eyeBlinkLeft`). Each expression detector in `expression_detector.py` reads relevant blendshapes and computes a composite confidence score:

```python
def detect_wink(blendshapes):
    blink_left  = get_blendshape(blendshapes, "eyeBlinkLeft")
    blink_right = get_blendshape(blendshapes, "eyeBlinkRight")
    max_blink = max(blink_left, blink_right)
    min_blink = min(blink_left, blink_right)
    if max_blink < 0.4 or min_blink > 0.15:
        return 0.0
    return clamp(max_blink - min_blink)  # asymmetry score
```

### 3. Gesture Stabilization Pipeline

Raw per-frame detections are noisy. A three-stage buffer smooths transitions:

```
Raw detection → [6-frame confirm buffer] → [40-frame hold lock] → [15-frame cooldown]
```

- **Confirm**: A gesture must appear in 6 consecutive frames before triggering
- **Hold**: Once triggered, the display holds for 40 frames regardless of new detections
- **Cooldown**: After returning to idle, ignores new gestures for 15 frames

---

## 📄 License

This project is open source. Feel free to use, modify, and distribute.

---

<p align="center">Built with ❤️ using Google MediaPipe &amp; OpenCV</p>