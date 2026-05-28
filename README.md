# 🎬 AI Video Clipper to Shorts Converter ⚡

[![Python Version](https://img.shields.io/badge/Python-3.9+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![AI Engine](https://img.shields.io/badge/Gemini_1.5_Flash-orange?style=for-the-badge&logo=google-gemini&logoColor=white)](https://ai.google.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

An advanced, open-source Python application designed to automatically extract engaging, high-energy highlights from long-form videos (up to 20-30 minutes) and convert them into viral vertical clips (9:16) optimized for TikTok, YouTube Shorts, and Instagram Reels. 

The application utilizes **local open-source AI models** for transcription, **Google Gemini** for intelligent highlight detection, and a custom high-performance frame-processing pipeline backed by **FFmpeg** and **OpenCV** to produce premium-grade social media content entirely for free!

---

## ✨ Premium Features

### 🤖 Dual-Mode Transcription Engine
*   **Cloud Gemini Mode (Recommended):** Leverages the Google Gemini Audio API to upload the audio stream. Transcribes and identifies viral highlights in **under 30 seconds** for a 30-minute video completely for free!
*   **Local Whisper Mode:** Runs fully offline using `faster-whisper` with automatic GPU (NVIDIA CUDA) detection to accelerate transcription speeds, falling back safely to CPU if CUDA is unavailable.

### 📐 4 Advanced Cropping & Reframing Modes
*   **Smart Auto-Reframing (Face Tracking):** Uses MediaPipe and OpenCV face-detection graphs to precompute facial centers, smoothing out raw movements with an **Exponential Moving Average (EMA)** filter to generate cinematic camera sweeps.
*   **Blurred Background (Modern Reel Style):** Fits the entire original horizontal 16:9 screen in the center of the vertical frame, filling the top and bottom voids with a heavily blurred, high-contrast backdrop of the same footage. **Perfect for gameplay clips!**
*   **Fit Entire Screen (Letterbox):** Scales down the original video to fit the vertical frame with clean, distraction-free black bars.
*   **Static Center Crop:** Traditional static crop focusing on the middle of the canvas.

### 📝 Dynamic Subtitle Engine
*   **Pure-Python outlined subtitles:** Burns high-energy outlined captions (neon yellow text with thick black outline) onto speaking highlights.
*   **Built-in baked subtitle safe:** Detects if a clip contains no spoken words (e.g., pure gameplay/silent montages) and safely suspends subtitles to prevent messy text overlaps on clips that already have in-game subtitles!

### 🔊 Automated Sound Peak Detection
*   If a video has no speech (Whisper/Gemini finds no transcript), the app automatically triggers a **Local Wave RMS Audio Energy Analysis** to find the 3 loudest, most action-packed highlights (explosions, weapon clashes, combat music swells) so gameplays are never skipped!

---

## 🛠️ Technology Stack
*   **Core UI:** [Streamlit](https://streamlit.io/) (Premium custom-styled dark theme)
*   **AI Transcription:** [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) (Local engine)
*   **LLM Reasoner:** [Google Gemini API](https://ai.google.dev/) (Cloud engine)
*   **Frame Processing:** [MoviePy](https://zulko.github.io/moviepy/) (Fully compatible with both v1.x and v2.x)
*   **Computer Vision:** [OpenCV](https://opencv.org/) & [MediaPipe](https://google.github.io/mediapipe/)
*   **Graphic Rendering:** [Pillow (PIL)](https://python-pillow.org/)
*   **Audio Engine:** [FFmpeg](https://ffmpeg.org/) (High-performance subprocess extraction)

---

## 💻 Web GUI Walkthrough

The application features a beautifully customized dark glassmorphic dashboard split into three clean control sections:

1.  **Sidebar Configuration Panel:**
    *   **Gemini API Key:** Input field with automated whitespace trimming and a **built-in typo warning system** (checks for typical Google API key typos like `AlzaSy` instead of `AIzaSy`).
    *   **Engine Selector:** Toggle between Cloud Gemini and Local Whisper.
    *   **Cropping Mode:** Dropdown to select your framing (Face Tracking, Blurred Background, Letterbox, Center Crop).
    *   **Burn Subtitles:** Toggle dynamic subtitles overlay.
2.  **Main Upload Dashboard:**
    *   Simple drag-and-drop file uploader supporting MP4, MOV, and MKV formats.
    *   Real-time progress pipeline showing the status of each step:
        *   `Step 1: Audio Extraction` (using FFmpeg stream copying).
        *   `Step 2: Transcription` (via Gemini Cloud or Whisper).
        *   `Step 3: Highlight Detection` (analyzing speech and sound energy).
3.  **Interactive Highlight Gallery:**
    *   Each detected highlight displays in its own card detailing the timeframe, exact duration, and Gemini's explanation for why the clip is viral-worthy.
    *   **📝 Transcript Preview:** Shows the full spoken text on the webpage next to the video, allowing the clip itself to remain clean!
    *   **⏱️ 7-Sec Preview Button:** Renders a 7-second vertical sample in less than 3 seconds, letting you inspect crop scaling before committing to a full render.
    *   **✂️ Render Full Clip Button:** Slices, reframes, and renders the entire vertical short.
    *   **⬇️ Standardized Download Button:** Saves the clip to your computer using a clean, professional naming standard: `VideoClipper_YYYYMMDD_XX.mp4` (sequential indexing).

---

## 🛠️ Setup & Installation

### Prerequisites

1.  **Python 3.9+** must be installed on your system.
2.  **FFmpeg** must be installed and added to your system's PATH variables.
    *   **Windows:** Run `winget install ffmpeg` in PowerShell, or download from [Gyan.dev](https://www.gyan.dev/ffmpeg/builds/).
    *   **macOS:** Run `brew install ffmpeg`.
    *   **Linux:** Run `sudo apt install ffmpeg`.

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/VideoClipper.git
cd VideoClipper
```

### 2. Create and Activate a Virtual Environment
```bash
python -m venv venv

# Windows (PowerShell)
.\venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup API Keys
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=AIzaSyYourGeminiAPIKeyHere
```
*(Note: When you enter your API key in the Web UI sidebar, it is automatically written to this `.env` file for you, so you only have to do it once!)*

---

## 🚀 Running the App

With your virtual environment activated, run:
```bash
streamlit run app.py
```
Open the local URL in your browser (usually `http://localhost:8501`).

---

## 🎮 Creator Pro Tips for Ultimate Quality

Our processing pipeline is highly optimized, but vertical upscaling of horizontal gameplay recordings depends heavily on source quality.

*   **Pristine Resolution (Lanczos4 Upscaling):** Our code implements professional **Lanczos4 foreground resizing** and **Bicubic background interpolation** inside OpenCV. This guarantees extremely sharp details and UI text, even on compressed videos.
*   **OBS Recording Settings:** For pristine results, record gameplays in OBS Studio in at least **1080p** (or 1440p) using a high bitrate (**12,000 - 20,000 Kbps**) with the **NVIDIA NVENC H.264** hardware encoder.
*   **Gameplay HUDs:** When converting action games, use the **`Blurred Background`** cropping mode in the sidebar. This ensures your viewer can see the entire screen (including weapon slots, health bars, and maps) in the center, rather than cutting the sides off!

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
