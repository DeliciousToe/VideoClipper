import streamlit as st
import os
import tempfile
import shutil
from dotenv import load_dotenv
from ai_utils import extract_audio, transcribe_audio, analyze_transcript_with_gemini, analyze_video_cloud
from video_utils import cut_and_crop_video

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="AI Video Clipper",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Custom CSS Styles
st.markdown("""
<style>
    /* Dark Theme Core Adjustments */
    .reportview-container {
        background: #0e1117;
    }
    
    /* Elegant Title and Header styling */
    h1 {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem !important;
    }
    
    /* Card design for Highlights */
    .highlight-card {
        background-color: #1e222b;
        border: 1px solid #2d3139;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 16px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .highlight-card:hover {
        transform: translateY(-2px);
        border-color: #FF4B4B;
    }
    
    /* Status indicator list style */
    .status-item {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        background-color: #161a22;
        border: 1px solid #242933;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    .status-active {
        border-color: #FF4B4B;
        background-color: rgba(255, 75, 75, 0.05);
        color: #FF8E53;
    }
    .status-done {
        border-color: #2ea043;
        background-color: rgba(46, 160, 67, 0.05);
        color: #56d364;
    }
    .status-pending {
        color: #8b949e;
    }
    
    /* Custom premium primary buttons with neon gradient */
    div.stButton > button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        letter-spacing: 0.5px;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.2);
        width: 100%;
    }
    div.stButton > button[data-testid="baseButton-primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(255, 75, 75, 0.4);
        background: linear-gradient(135deg, #ff5e5e 0%, #ffa473 100%);
    }
    div.stButton > button[data-testid="baseButton-primary"]:active {
        transform: translateY(1px);
    }
    
    /* Secondary actions (darker style) perfectly aligned */
    div.stButton > button[data-testid="baseButton-secondary"] {
        background: #21262d !important;
        color: #c9d1d9 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1) !important;
        width: 100%;
    }
    div.stButton > button[data-testid="baseButton-secondary"]:hover {
        background: #30363d !important;
        border-color: #8b949e !important;
        transform: translateY(-2px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 22px;">
        <div style="background: linear-gradient(135deg, #FF4B4B 0%, #FF8E53 100%); padding: 12px; border-radius: 12px; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.3); display: flex; align-items: center; justify-content: center;">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                <rect x="2" y="2" width="20" height="20" rx="2.18" ry="2.18"></rect>
                <line x1="7" y1="2" x2="7" y2="22"></line>
                <line x1="17" y1="2" x2="17" y2="22"></line>
                <line x1="2" y1="12" x2="22" y2="12"></line>
                <line x1="2" y1="7" x2="7" y2="7"></line>
                <line x1="2" y1="17" x2="7" y2="17"></line>
                <line x1="17" y1="17" x2="22" y2="17"></line>
                <line x1="17" y1="7" x2="22" y2="7"></line>
            </svg>
        </div>
        <span style="font-size: 22px; font-weight: 800; color: white; font-family: 'Outfit', 'Inter', sans-serif; letter-spacing: 0.5px; line-height: 1;">VideoClipper</span>
    </div>
    """, unsafe_allow_html=True)
    st.header("⚙️ Configuration")
    
    api_key = st.text_input(
        "Gemini API Key",
        value=os.getenv("GEMINI_API_KEY", ""),
        type="password",
        help="Required for Cloud Mode and Local Mode AI highlight analysis."
    )
    
    # Save key dynamically to .env so it retains after reload!
    if api_key:
        with open(".env", "w") as f:
            f.write(f"GEMINI_API_KEY={api_key.strip()}\n")
        os.environ["GEMINI_API_KEY"] = api_key.strip()
    
    # Check for common typo in Gemini API Key (AlzaSy instead of AIzaSy)
    if api_key.startswith("AlzaSy"):
        st.warning("⚠️ Typo Warning: Key starts with 'AlzaSy' (lowercase L). Google API Keys start with 'AIzaSy' (uppercase I). Please verify.")
    
    st.subheader("🤖 Transcription Engine")
    engine = st.radio(
        "Select Engine",
        ["Cloud Gemini (Recommended)", "Local Whisper"],
        index=0,
        help="Cloud Gemini is ultra fast (< 30 seconds for 30 min audio) and highly accurate. Local Whisper is fully offline but runs on your computer."
    )
    
    if engine == "Local Whisper":
        model_size = st.selectbox(
            "Whisper Model Size",
            ["tiny", "base", "small", "medium"],
            index=1,
            help="Larger models are more accurate but consume more RAM and CPU/GPU time."
        )
    else:
        model_size = "base" # Default placeholder
        
    st.subheader("✂️ Cutting Settings")
    crop_mode = st.selectbox(
        "Cropping Mode / Reframing",
        ["Smart Face Tracking", "Fit Entire Screen (Letterbox)", "Blurred Background", "Center Crop"],
        index=0,
        help="Choose how to frame the vertical 9:16 output. 'Smart Face Tracking' keeps speakers centered. 'Blurred Background' and 'Letterbox' fit the entire horizontal screen, ideal for gameplays!"
    )
    
    enable_subtitles = st.checkbox(
        "Burn Dynamic Subtitles",
        value=True,
        help="Overlay clean, high-impact captions on the vertical video (no ImageMagick required)."
    )
    
    st.markdown("---")
    st.markdown("🌐 **Fully Open-Source & Freeware**")
    st.caption("Powered by Streamlit, FFmpeg, Pillow, MediaPipe, Faster-Whisper & Gemini.")

# ----------------- MAIN LAYOUT -----------------
st.title("🎬 AI Video Clipper to Shorts")
st.markdown("Convert your long videos (up to 30 mins) into vertical viral reels and TikTok highlights in seconds!")
st.markdown("---")

# Initialize session state variables to cache processed results
if 'temp_dir' not in st.session_state:
    st.session_state.temp_dir = tempfile.mkdtemp()
if 'video_path' not in st.session_state:
    st.session_state.video_path = None
if 'audio_path' not in st.session_state:
    st.session_state.audio_path = None
if 'highlights' not in st.session_state:
    st.session_state.highlights = []
if 'previews' not in st.session_state:
    st.session_state.previews = {}
if 'clips' not in st.session_state:
    st.session_state.clips = {}

# File Uploader
uploaded_file = st.file_uploader(
    "📥 Drag and drop your video file here (MP4, MOV, MKV)",
    type=["mp4", "mov", "mkv"],
    help="Supports up to 20-30 minute long videos."
)

if uploaded_file is not None:
    # Check if a new file was uploaded
    current_filename = os.path.join(st.session_state.temp_dir, uploaded_file.name)
    if st.session_state.video_path != current_filename:
        # Reset cached states for new files
        st.session_state.video_path = current_filename
        st.session_state.audio_path = os.path.join(st.session_state.temp_dir, "audio.wav")
        st.session_state.highlights = []
        st.session_state.previews = {}
        st.session_state.clips = {}
        
        # Save uploaded file
        with open(st.session_state.video_path, "wb") as f:
            f.write(uploaded_file.read())
            
    # Layout splits: Original video on left, Analysis dashboard on right
    col_vid, col_panel = st.columns([1, 1.2])
    
    with col_vid:
        st.subheader("🎥 Original Video")
        st.video(st.session_state.video_path)
        st.caption(f"📁 {uploaded_file.name} | Loaded successfully.")
        
    with col_panel:
        st.subheader("⚡ Clipper Control Panel")
        st.write("Configure settings in the sidebar and trigger the analysis below.")
        
        # Generate Highlights button
        if st.button("🚀 Analyze & Extract Highlights", type="primary"):
            if not api_key:
                st.error("⚠️ Please enter a valid Gemini API Key in the sidebar.")
            else:
                # Progress Pipeline representation
                status_block = st.empty()
                
                def render_status(step1="pending", step2="pending", step3="pending"):
                    icons = {"pending": "⚪", "active": "⏳", "done": "✅"}
                    classes = {"pending": "status-pending", "active": "status-active", "done": "status-done"}
                    
                    status_block.markdown(f"""
                    <div style="margin-bottom: 20px;">
                        <div class="status-item {classes[step1]}">
                            <span>{icons[step1]}</span> <b>Step 1:</b> High-Performance Audio Extraction
                        </div>
                        <div class="status-item {classes[step2]}">
                            <span>{icons[step2]}</span> <b>Step 2:</b> Audio Transcription ({engine})
                        </div>
                        <div class="status-item {classes[step3]}">
                            <span>{icons[step3]}</span> <b>Step 3:</b> AI Viral Highlight Detection
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Execution pipeline
                try:
                    # Step 1: Audio Extraction
                    render_status("active", "pending", "pending")
                    extract_audio(st.session_state.video_path, st.session_state.audio_path)
                    
                    # Step 2 & 3: Cloud Mode vs Local Mode
                    if "Cloud" in engine:
                        render_status("done", "active", "pending")
                        highlights = analyze_video_cloud(st.session_state.audio_path, api_key)
                        
                        # Fallback for silent / non-voice gameplay recordings
                        if not highlights:
                            st.info("💡 Cloud analysis returned no speech highlights. Switching to Local Audio Energy Analysis to detect action peaks...")
                            from video_utils import analyze_audio_energy_local
                            highlights = analyze_audio_energy_local(st.session_state.audio_path)
                            
                        render_status("done", "done", "done")
                    else:
                        # Local Mode
                        render_status("done", "active", "pending")
                        transcript = transcribe_audio(st.session_state.audio_path, model_size=model_size)
                        
                        # Fallback for silent / non-voice gameplay recordings
                        if not transcript:
                            st.info("💡 Local Whisper detected no spoken words. Switching to Local Audio Energy Analysis to detect action peaks...")
                            from video_utils import analyze_audio_energy_local
                            highlights = analyze_audio_energy_local(st.session_state.audio_path)
                            render_status("done", "done", "done")
                        else:
                            render_status("done", "done", "active")
                            highlights = analyze_transcript_with_gemini(transcript, api_key)
                            render_status("done", "done", "done")
                        
                    if highlights:
                        st.session_state.highlights = highlights
                        st.success(f"✨ Successfully found {len(highlights)} viral highlights!")
                    else:
                        st.error("❌ Failed to detect highlights. Please verify your Gemini API key (note any typo warnings in the sidebar), or check the video audio track.")
                        
                except Exception as e:
                    st.error(f"Error during video processing: {e}")
                    
        # If highlights already exist in state, show summary stats
        if st.session_state.highlights:
            st.markdown(f"**⚡ Loaded Highlights:** {len(st.session_state.highlights)} clips identified.")

# ----------------- HIGHLIGHTS & CLIPS DISPLAY -----------------
if st.session_state.highlights:
    st.markdown("---")
    st.subheader("🔥 Detected Highlight Clips")
    st.markdown("Select from the highlights below. You can **Preview** a 7-second sample or **Render** the full vertical 9:16 short.")
    
    for i, highlight in enumerate(st.session_state.highlights):
        title = highlight.get("title", f"Highlight #{i+1}")
        explanation = highlight.get("explanation", "")
        start_time = highlight.get("start_time", 0.0)
        end_time = highlight.get("end_time", 0.0)
        duration = end_time - start_time
        
        # Action controls inside streamlit columns (Info card is now inside col_actions on the left!)
        col_actions, col_media = st.columns([1, 1])
        
        with col_actions:
            # Display clip card on the left panel
            st.markdown(f"""
            <div class="highlight-card" style="padding: 18px; margin-bottom: 14px;">
                <h3 style="margin-top:0; color:#FF8E53; font-size: 19px; margin-bottom: 8px;">🎬 Clip {i+1}: {title}</h3>
                <p style="font-style: italic; color: #c9d1d9; font-size: 13.5px; margin-bottom: 8px;">"{explanation}"</p>
                <div style="font-size: 12.5px; color: #8b949e; line-height: 1.4;">
                    ⏱️ <b>Timeframe:</b> {start_time:.1f}s - {end_time:.1f}s<br>
                    📐 <b>Duration:</b> {duration:.1f} seconds
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show transcript preview if subtitles are present
            sub_list = highlight.get("subtitles", [])
            if sub_list:
                transcript_text = " ".join([s["text"] for s in sub_list if s.get("text")])
                if transcript_text.strip():
                    st.markdown(f"""
                    <div style="background-color: #161a22; border: 1px solid #242933; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
                        <span style="font-size: 11px; font-weight: 700; color: #FF8E53; text-transform: uppercase; letter-spacing: 0.5px;">📝 Transcript Preview</span>
                        <p style="font-size: 13px; color: #c9d1d9; margin: 4px 0 0 0; line-height: 1.4;">"{transcript_text}"</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            st.write("⚙️ **Clip Operations:**")
            
            # Action buttons
            col_btn1, col_btn2 = st.columns(2)
            
            with col_btn1:
                preview_clicked = st.button(f"⏱️ 7-Sec Preview", key=f"prev_btn_{i}", type="secondary")
                
            with col_btn2:
                render_clicked = st.button(f"✂️ Render Full Clip", key=f"render_btn_{i}", type="primary")
                
            # Process Clip triggers: Do NOT pass subtitles if they are empty
            subtitles = highlight.get("subtitles", []) if enable_subtitles else None
            if not subtitles:
                subtitles = None
            
            # 1. Handle Preview Request
            if preview_clicked:
                preview_path = os.path.join(st.session_state.temp_dir, f"clip_{i+1}_preview.mp4")
                # Define 7 second limit for preview
                preview_end = min(start_time + 7.0, end_time)
                
                with st.spinner("Generating 7-second sample..."):
                    try:
                        cut_and_crop_video(
                            st.session_state.video_path,
                            start_time,
                            preview_end,
                            preview_path,
                            subtitles=subtitles,
                            crop_mode=crop_mode
                        )
                        st.session_state.previews[i] = preview_path
                        st.success("Preview generated successfully!")
                    except Exception as e:
                        st.error(f"Failed to generate preview: {e}")
                        
            # 2. Handle Render Full Request
            if render_clicked:
                clip_path = os.path.join(st.session_state.temp_dir, f"clip_{i+1}_full.mp4")
                
                with st.spinner("Rendering full vertical clip (this processes every frame)..."):
                    try:
                        cut_and_crop_video(
                            st.session_state.video_path,
                            start_time,
                            end_time,
                            clip_path,
                            subtitles=subtitles,
                            crop_mode=crop_mode
                        )
                        st.session_state.clips[i] = clip_path
                        st.success("Full vertical clip rendered successfully!")
                    except Exception as e:
                        st.error(f"Failed to render full clip: {e}")
                        
        with col_media:
            # Render and display results if they exist in state
            has_media = False
            
            if i in st.session_state.clips:
                st.write("✅ **Full Rendered Short:**")
                st.video(st.session_state.clips[i])
                
                # Standardized file name format: VideoClipper_RRRRMMDDXX
                from datetime import datetime
                today_str = datetime.now().strftime("%Y%m%d")
                download_filename = f"VideoClipper_{today_str}_{i+1:02d}.mp4"
                
                # Download Button
                with open(st.session_state.clips[i], "rb") as f:
                    st.download_button(
                        label="⬇️ Download Full Short (9:16)",
                        data=f,
                        file_name=download_filename,
                        mime="video/mp4",
                        key=f"dl_full_{i}"
                    )
                has_media = True
                
            elif i in st.session_state.previews:
                st.write("⏱️ **Preview Short (First 5 seconds):**")
                st.video(st.session_state.previews[i])
                st.info("💡 Like the preview? Click **Render Full Clip** to get the full video!")
                has_media = True
                
            if not has_media:
                st.info("ℹ️ No video rendered yet. Click **5-Sec Preview** to test styles or **Render Full Clip** to generate the final vertical video.")
                
        st.markdown("---")
