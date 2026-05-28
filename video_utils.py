import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
try:
    from moviepy import VideoFileClip
except ImportError:
    from moviepy.editor import VideoFileClip

def get_font(font_size):
    """
    Loads a clean, highly readable bold font from the Windows system fonts directory.
    Falls back gracefully to standard Arial or the PIL default font if not found.
    """
    font_paths = [
        "C:\\Windows\\Fonts\\impact.ttf",      # Standard high-energy meme/reel font
        "C:\\Windows\\Fonts\\arialbd.ttf",     # Arial Bold
        "C:\\Windows\\Fonts\\segoeui.ttf",     # Segoe UI Bold
        "C:\\Windows\\Fonts\\arial.ttf"
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    try:
        return ImageFont.load_default()
    except Exception:
        return None

def precompute_face_centers(video_path, start_time, end_time, sample_fps=5):
    """
    Precomputes the coordinates of the speaker's face X center at a lower FPS
    (e.g., 5 frames per second) to keep execution incredibly fast and efficient.
    Interpolates between coordinates during rendering to achieve buttery smooth cinematic panning.
    """
    face_detector = None
    try:
        import mediapipe as mp
        # Initialize MediaPipe Face Detection
        mp_face_detection = mp.solutions.face_detection
        face_detector = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.45)
        print("Initialized MediaPipe for high-accuracy face tracking.")
    except Exception as e:
        print(f"MediaPipe failed to load, falling back to OpenCV Haar Cascade: {e}")
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            face_detector = cv2.CascadeClassifier(cascade_path)
            print("Initialized OpenCV Haar Cascade for face tracking.")
        except Exception as cascade_err:
            print(f"Face tracking unavailable (OpenCV Haar Cascade failed to load): {cascade_err}")
            face_detector = None

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file for face detection at {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Calculate crop dimensions to ensure bounds
    target_ratio = 9 / 16.0
    crop_w = int(height * target_ratio)
    if crop_w > width:
        crop_w = width
    min_x = crop_w // 2
    max_x = width - (crop_w // 2)

    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    sample_interval = max(1, int(fps / sample_fps))
    
    detected_centers = []

    for frame_idx in range(start_frame, end_frame, sample_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        t = frame_idx / fps
        face_x = None

        if face_detector is not None:
            # 1. MediaPipe Face Detection
            if hasattr(face_detector, 'process'):
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = face_detector.process(rgb_frame)
                if results.detections:
                    detection = results.detections[0]
                    bbox = detection.location_data.relative_bounding_box
                    face_center_rel_x = bbox.xmin + (bbox.width / 2.0)
                    face_x = int(face_center_rel_x * width)
            # 2. OpenCV Haar Cascade Fallback
            else:
                if hasattr(face_detector, 'empty') and not face_detector.empty():
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(60, 60))
                    if len(faces) > 0:
                        # Choose the largest face
                        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                        fx, fy, fw, fh = faces[0]
                        face_x = fx + (fw // 2)
                else:
                    face_x = None

        if face_x is not None:
            # Clamp the face coordinate to keep the 9:16 crop window perfectly inside the 16:9 canvas
            face_x = max(min_x, min(face_x, max_x))
            detected_centers.append((t, face_x))

    cap.release()

    if not detected_centers:
        print("No faces detected in this clip. Defaulting to center crop.")
        return []

    # Apply Exponential Moving Average (EMA) to smooth the camera panning and prevent jerky motions
    alpha = 0.12  # Smoothing factor (lower is smoother, higher follows face faster)
    smoothed_centers = []
    current_val = detected_centers[0][1]
    
    for t, val in detected_centers:
        current_val = alpha * val + (1 - alpha) * current_val
        smoothed_centers.append((t, current_val))

    return smoothed_centers

def get_crop_center(t, smoothed_centers, default_center):
    """
    Linearly interpolates the precomputed, smoothed face centers to get the exact crop X coordinate
    for the exact frame timestamp t.
    """
    if not smoothed_centers:
        return default_center

    times = [sc[0] for sc in smoothed_centers]
    coords = [sc[1] for sc in smoothed_centers]

    if t <= times[0]:
        return int(coords[0])
    if t >= times[-1]:
        return int(coords[-1])

    # Interpolate coordinate
    return int(np.interp(t, times, coords))

def cut_and_crop_video(video_path, start_time, end_time, output_path, subtitles=None, crop_mode="Smart Face Tracking"):
    """
    Cuts a video segment, performs automatic AI face reframing or screen fitting, overlays dynamic word-aligned
    Pillow-drawn subtitles, and renders a premium vertical 9:16 short.
    """
    enable_face_tracking = (crop_mode == "Smart Face Tracking")
    
    # 1. Precompute smoothed face center coordinates if tracking is active
    smoothed_centers = []
    if enable_face_tracking:
        try:
            smoothed_centers = precompute_face_centers(video_path, start_time, end_time)
        except Exception as e:
            print(f"Failed face tracking computation: {e}. Defaulting to center crop.")
            smoothed_centers = []

    # 2. Open video clip and apply custom frame processor
    with VideoFileClip(video_path) as video:
        w, h = video.size
        target_ratio = 9 / 16.0
        
        # Calculate optimal vertical crop size
        crop_h = h
        crop_w = int(h * target_ratio)
        
        # In case video is ultra-tall (unlikely)
        if crop_w > w:
            crop_w = w
            crop_h = int(w / target_ratio)

        default_center_x = w // 2
        
        # Clamp start_time and end_time to original video duration boundaries to prevent crashes
        vid_duration = video.duration if hasattr(video, "duration") else h / 30.0 # fallback
        start_time = max(0.0, min(start_time, vid_duration))
        end_time = max(0.0, min(end_time, vid_duration))
        
        # Slice the subclip (compatibility with MoviePy 1.x and 2.x)
        if hasattr(video, "subclipped"):
            clip = video.subclipped(start_time, end_time)
        else:
            clip = video.subclip(start_time, end_time)

        # Precompute resize and padding coordinates for Letterbox / Blur modes
        scale = crop_w / float(w)
        scaled_w = crop_w
        scaled_h = int(h * scale)
        pad_top = (crop_h - scaled_h) // 2
        
        # Precompute background scale for blurred backdrop
        bg_scale = crop_h / float(h)
        bg_w = int(w * bg_scale)
        bg_x1 = (bg_w - crop_w) // 2

        # Custom frame processor applying crop & subtitle overlays
        def process_frame(get_frame, t):
            frame = get_frame(t)
            abs_t = start_time + t
            
            # --- CROPPING MODES ---
            if crop_mode == "Fit Entire Screen (Letterbox)":
                # Rescale with Lanczos4 interpolation for crystal-clear, sharp gameplay details
                scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
                canvas = np.zeros((crop_h, crop_w, 3), dtype=np.uint8)
                canvas[pad_top:pad_top+scaled_h, :] = scaled_frame
                cropped_frame = canvas
                
            elif crop_mode == "Blurred Background":
                # 1. Scale background using Bicubic interpolation for high-quality upscaling
                bg_resized = cv2.resize(frame, (bg_w, crop_h), interpolation=cv2.INTER_CUBIC)
                bg_cropped = bg_resized[:, bg_x1:bg_x1+crop_w]
                # 2. Apply Gaussian blur to the background
                bg_blurred = cv2.GaussianBlur(bg_cropped, (51, 51), 0)
                # 3. Scale actual foreground frame with Lanczos4 for professional gameplay sharpness
                scaled_frame = cv2.resize(frame, (scaled_w, scaled_h), interpolation=cv2.INTER_LANCZOS4)
                bg_blurred[pad_top:pad_top+scaled_h, :] = scaled_frame
                cropped_frame = bg_blurred
                
            else:
                # Face Tracking or Static Center Crop
                center_x = default_center_x
                if enable_face_tracking and smoothed_centers:
                    center_x = get_crop_center(abs_t, smoothed_centers, default_center_x)

                x1 = max(0, center_x - (crop_w // 2))
                x2 = min(w, x1 + crop_w)
                y1 = (h - crop_h) // 2
                y2 = y1 + crop_h
                cropped_frame = frame[y1:y2, x1:x2]

            # Subtitle Burning
            if subtitles:
                current_text = None
                for sub in subtitles:
                    if sub["start"] <= abs_t <= sub["end"]:
                        current_text = sub["text"]
                        break

                if current_text:
                    img = Image.fromarray(cropped_frame)
                    draw = ImageDraw.Draw(img)
                    cw, ch = img.size
                    
                    # Compute responsive font size based on video height (approx 5.5% of height)
                    font_size = int(ch * 0.055)
                    font = get_font(font_size)
                    
                    text = current_text.upper().strip()
                    
                    # Calculate centered position
                    try:
                        bbox = draw.textbbox((0, 0), text, font=font)
                        tw = bbox[2] - bbox[0]
                        th = bbox[3] - bbox[1]
                    except AttributeError:
                        tw, th = draw.textsize(text, font=font)

                    tx = (cw - tw) // 2
                    ty = int(ch * 0.78)  # Position at 78% height for reels standard

                    # Draw thick black outline with bright high-visibility yellow text
                    outline_color = (0, 0, 0)
                    text_color = (255, 235, 59)  # Highly visible bright viral yellow
                    stroke_width = max(2, int(font_size * 0.08))

                    draw.text(
                        (tx, ty), text,
                        font=font,
                        fill=text_color,
                        stroke_width=stroke_width,
                        stroke_fill=outline_color
                    )
                    cropped_frame = np.array(img)

            return cropped_frame

        # Apply processor to each frame in real time (compatibility with MoviePy 1.x and 2.x)
        if hasattr(clip, "transform"):
            processed_clip = clip.transform(process_frame)
        else:
            processed_clip = clip.fl(process_frame)

        # Write output file with high compatibility options
        processed_clip.write_videofile(
            output_path,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=os.path.join(os.path.dirname(output_path), "temp-audio.m4a"),
            remove_temp=True,
            logger=None
        )

    return output_path

def analyze_audio_energy_local(audio_path):
    """
    Analyzes the audio WAV file's amplitude (energy) to find the 3 loudest/most energetic
    segments of 20-30 seconds. This is the perfect fallback for videos without speech (like gameplay).
    """
    import wave
    import struct
    
    try:
        wf = wave.open(audio_path, 'rb')
        params = wf.getparams()
        nchannels, sampwidth, framerate, nframes = params[:4]
        
        # Read chunks of 1 second
        chunk_size = framerate
        bytes_per_sample = sampwidth * nchannels
        
        energies = []
        
        for i in range(0, nframes, chunk_size):
            data = wf.readframes(chunk_size)
            if not data:
                break
            
            # Compute Root Mean Square (RMS) energy
            # If 16-bit audio (standard WAV from FFmpeg output)
            if sampwidth == 2:
                fmt = f"<{len(data)//2}h"
                samples = struct.unpack(fmt, data)
                # RMS
                sq = [s**2 for s in samples[::12]] # Subsample slightly to keep it fast
                rms = (sum(sq) / len(sq)) ** 0.5 if sq else 0
                energies.append(rms)
            else:
                energies.append(0)
        wf.close()
        
        # If no energies are calculated, fallback
        if not energies:
            return []
            
        # Smooth energies with moving average
        smoothed = []
        window = 15 # 15 seconds window
        for i in range(len(energies)):
            start_idx = max(0, i - window // 2)
            end_idx = min(len(energies), i + window // 2)
            smoothed.append(sum(energies[start_idx:end_idx]) / (end_idx - start_idx))
            
        # Find top 3 non-overlapping peaks of 30s
        highlights = []
        duration = len(energies)
        
        # Sort indices by energy
        sorted_indices = sorted(range(len(smoothed)), key=lambda k: smoothed[k], reverse=True)
        
        for idx in sorted_indices:
            if len(highlights) >= 3:
                break
                
            start_time = max(0, idx - 15)
            end_time = min(duration, idx + 15)
            
            # Check overlap
            overlap = False
            for h in highlights:
                if not (end_time <= h["start_time"] or start_time >= h["end_time"]):
                    overlap = True
                    break
            
            if not overlap and (end_time - start_time) >= 10:
                highlights.append({
                    "title": f"Action Highlight #{len(highlights)+1}",
                    "start_time": float(start_time),
                    "end_time": float(end_time),
                    "explanation": "Detected high-energy action peak in audio recording.",
                    "subtitles": []
                })
        return highlights
    except Exception as e:
        print(f"Error in local energy analysis: {e}")
        return []

