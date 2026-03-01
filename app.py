"""
GenTA GACS Mini - Streamlit Frontend
Interactive dashboard for exploring mood/style similarity in videos
WITH AI-POWERED EXPLANATIONS
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import cv2
from PIL import Image
import os
import subprocess
import sys
import time
import yaml
import base64
import tempfile
from io import BytesIO
import shutil
import requests
import json
from sklearn.metrics.pairwise import cosine_similarity

# Page configuration
st.set_page_config(
    page_title="GenTA GACS Mini",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2196F3;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .score-high {
        color: #4CAF50;
        font-weight: bold;
    }
    .score-medium {
        color: #FF9800;
        font-weight: bold;
    }
    .score-low {
        color: #f44336;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
    }
    .upload-box {
        border: 2px dashed #4CAF50;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
    }
    .explanation-box {
        background-color: #1E1E1E;
        color: #FFFFFF;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #4CAF50;
        margin-top: 1rem;
        font-family: 'Courier New', monospace;
        font-size: 1rem;
        line-height: 1.6;
    }
    .explanation-box h4 {
        color: #4CAF50;
        margin-top: 0;
        font-size: 1.2rem;
        margin-bottom: 1rem;
    }
    .insight-tag {
        background-color: #4CAF50;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
        margin-right: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'page' not in st.session_state:
    st.session_state.page = "Home"
if 'uploaded_videos' not in st.session_state:
    st.session_state.uploaded_videos = []
if 'config' not in st.session_state:
    # Default configuration with ALL required keys and correct types
    st.session_state.config = {
        'video_folder': 'videos',
        'frames_folder': 'frames',
        'frames_per_video': 30,
        'frame_rate': 1.0,
        'model_name': 'openai/clip-vit-base-patch32',
        'batch_size': 16,
        'embedding_dim': 512,
        'top_k': 5,
        'query_indices': [0, 15, 25]
    }
    
    # Save initial config to file
    with open('config.yaml', 'w') as f:
        yaml.dump(st.session_state.config, f, default_flow_style=False)
        
if 'pipeline_complete' not in st.session_state:
    st.session_state.pipeline_complete = False
if 'selected_query' not in st.session_state:
    st.session_state.selected_query = None
if 'last_explanation' not in st.session_state:
    st.session_state.last_explanation = None

# Create necessary directories
Path('videos').mkdir(exist_ok=True)
Path('frames').mkdir(exist_ok=True)
Path('outputs').mkdir(exist_ok=True)

# Function to get AI explanation using OpenRouter (free tier)
def get_ai_explanation(query_frame, similar_frames, scores):
    """
    Use OpenRouter API (free) to explain similarity results in plain English.
    """
    try:
        # Prepare the prompt
        prompt = f"""You are an AI art critic explaining why certain video frames are visually similar.

Query Frame: {query_frame}
Similar Frames and their similarity scores:
"""
        for i, (frame, score) in enumerate(zip(similar_frames, scores)):
            prompt += f"{i+1}. {frame} (score: {score:.3f})\n"
        
        prompt += """

Explain in simple, conversational English why these frames are considered similar. 
Focus on visual elements like colors, composition, lighting, mood, and style.
Be specific and insightful, like a knowledgeable art friend explaining what they notice.
Keep it to 3-4 sentences maximum."""

        # Make API request to OpenRouter (free tier)
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": "Bearer OPENROUTER_API_KEY_DEV",
                "Content-Type": "application/json",
            },
            json={
                "model": "mistralai/mistral-7b-instruct",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            return result['choices'][0]['message']['content']
        else:
            return f"✨ These frames share similar visual characteristics. The similarity scores range from {min(scores):.2f} to {max(scores):.2f}."
            
    except Exception as e:
        # Fallback explanation if API fails
        return f"✨ These frames appear visually similar based on composition and style. Scores: {min(scores):.2f} to {max(scores):.2f}."

# Function to ensure config has correct types
def ensure_config_types(config):
    """Ensure all config values have the correct type."""
    config['frames_per_video'] = int(config.get('frames_per_video', 30))
    config['frame_rate'] = float(config.get('frame_rate', 1.0))
    config['batch_size'] = int(config.get('batch_size', 16))
    config['top_k'] = int(config.get('top_k', 5))
    config['embedding_dim'] = int(config.get('embedding_dim', 512))
    
    # Ensure query_indices is a list of ints
    if isinstance(config.get('query_indices'), str):
        try:
            config['query_indices'] = [int(x.strip()) for x in config['query_indices'].split(',')]
        except:
            config['query_indices'] = [0, 15, 25]
    elif not isinstance(config.get('query_indices'), list):
        config['query_indices'] = [0, 15, 25]
    
    return config

# Apply type checking to existing config
if 'config' in st.session_state:
    st.session_state.config = ensure_config_types(st.session_state.config)

# Sidebar navigation
with st.sidebar:
    st.title("🎨 GenTA GACS")
    st.markdown("---")
    
    # Navigation buttons
    if st.button("🏠 Home", use_container_width=True):
        st.session_state.page = "Home"
        st.rerun()
    if st.button("📤 Upload Videos", use_container_width=True):
        st.session_state.page = "Upload"
        st.rerun()
    if st.button("⚙️ Configure", use_container_width=True):
        st.session_state.page = "Configure"
        st.rerun()
    if st.button("🚀 Run Pipeline", use_container_width=True):
        st.session_state.page = "Run"
        st.rerun()
    if st.button("🔍 Explore Results", use_container_width=True):
        st.session_state.page = "Explore"
        st.rerun()
    
    st.markdown("---")
    
    # Status panel
    st.markdown("### 📊 Status")
    
    video_count = len(list(Path('videos').glob('*.mp4')))
    frame_count = len(list(Path('frames').glob('*.jpg')))
    embeddings_exist = Path('embeddings.npy').exists()
    metadata_exist = Path('metadata.csv').exists()
    
    st.metric("Videos", video_count)
    st.metric("Frames Extracted", frame_count)
    st.metric("Embeddings", "✅ Ready" if embeddings_exist else "❌ Pending")
    st.metric("Metadata", "✅ Ready" if metadata_exist else "❌ Pending")
    
    # AI Explanation toggle
    st.markdown("---")
    st.markdown("### 🤖 AI Features")
    enable_ai = st.checkbox("Enable AI Explanations", value=True, 
                            help="Use AI to explain why frames are similar in plain English")

# Main content based on selected page
if st.session_state.page == "Home":
    st.markdown('<h1 class="main-header">GenTA GACS Mini</h1>', unsafe_allow_html=True)
    st.markdown('<h2 class="sub-header">Affective Computing for Art & Marketing Videos</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🎯 What is this?")
        st.info(
            """
            This tool analyzes the "feel" or "vibe" of art and marketing videos using AI.
            It extracts frames, computes CLIP embeddings, and finds visually similar frames
            across different videos.
            """
        )
        
        st.markdown("### 🔍 How it works")
        st.markdown("""
        1. **Upload** 2-3 short MP4 videos
        2. **Configure** extraction settings
        3. **Run** the pipeline to generate embeddings
        4. **Explore** similar frames with AI explanations
        """)
    
    with col2:
        st.markdown("### 📋 Quick Start")
        st.markdown("""
        **Step 1:** Click 'Upload Videos' in sidebar
        **Step 2:** Upload 2-3 MP4 files
        **Step 3:** Click 'Configure' to adjust settings
        **Step 4:** Click 'Run Pipeline' to process
        **Step 5:** Click 'Explore Results' to see similar frames with AI insights
        """)
        
        if video_count < 2:
            st.warning(f"⚠️ Need at least 2 videos. Currently have {video_count}")
        else:
            st.success(f"✅ {video_count} videos ready!")
        
        # Show quick status
        with st.expander("📊 Current Status"):
            st.write(f"**Videos:** {video_count}")
            st.write(f"**Frames:** {frame_count}")
            st.write(f"**Embeddings:** {'✅' if embeddings_exist else '❌'}")
            st.write(f"**Metadata:** {'✅' if metadata_exist else '❌'}")
            st.write(f"**AI Explanations:** {'✅ Enabled' if enable_ai else '❌ Disabled'}")

elif st.session_state.page == "Upload":
    st.markdown('<h1 class="main-header">📤 Upload Videos</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Select MP4 Videos")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose 2-3 MP4 videos",
            type=['mp4'],
            accept_multiple_files=True,
            help="Upload short art gallery tours, abstract animations, or marketing videos"
        )
        
        if uploaded_files:
            if len(uploaded_files) < 2:
                st.warning("Please upload at least 2 videos")
            else:
                if st.button("💾 Save Videos", type="primary", use_container_width=True):
                    with st.spinner("Saving videos..."):
                        # Clear existing videos
                        for f in Path('videos').glob('*.mp4'):
                            f.unlink()
                        
                        # Save new videos
                        for uploaded_file in uploaded_files:
                            file_path = Path('videos') / uploaded_file.name
                            with open(file_path, 'wb') as f:
                                f.write(uploaded_file.getbuffer())
                        
                        st.success(f"✅ Saved {len(uploaded_files)} videos successfully!")
                        st.session_state.uploaded_videos = uploaded_files
                        
                        # Show saved videos
                        st.markdown("### 📋 Saved Videos:")
                        for f in Path('videos').glob('*.mp4'):
                            file_size = f.stat().st_size / (1024 * 1024)  # MB
                            st.text(f"📹 {f.name} ({file_size:.1f} MB)")
    
    with col2:
        st.markdown("### 📋 Current Videos")
        video_files = list(Path('videos').glob('*.mp4'))
        
        if video_files:
            for v in video_files:
                st.text(f"📹 {v.name}")
            
            if st.button("🗑️ Clear All", use_container_width=True):
                for f in Path('videos').glob('*.mp4'):
                    f.unlink()
                st.rerun()
        else:
            st.info("No videos uploaded")
        
        st.markdown("### 📝 Requirements")
        st.markdown("""
        - **Format**: MP4
        - **Count**: 2-3 videos
        - **Duration**: 15-60 seconds
        - **Content**: Art, abstract, marketing
        """)

elif st.session_state.page == "Configure":
    st.markdown('<h1 class="main-header">⚙️ Configure Pipeline</h1>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Extraction Settings")
        
        frames_per_video = st.slider(
            "Frames per video",
            min_value=10,
            max_value=100,
            value=int(st.session_state.config['frames_per_video']),
            step=5,
            help="Maximum number of frames to extract from each video"
        )
        
        # CRITICAL FIX: Ensure frame_rate is float
        current_frame_rate = st.session_state.config.get('frame_rate', 1.0)
        if isinstance(current_frame_rate, int):
            current_frame_rate = float(current_frame_rate)
            
        frame_rate = st.slider(
            "Frame rate (fps)",
            min_value=0.5,
            max_value=5.0,
            value=float(current_frame_rate),
            step=0.5,
            help="Frames to extract per second of video"
        )
    
    with col2:
        st.markdown("### 🤖 Model Settings")
        
        batch_size = st.slider(
            "Batch size",
            min_value=4,
            max_value=32,
            value=int(st.session_state.config['batch_size']),
            step=4,
            help="Lower if you run out of memory"
        )
        
        top_k = st.slider(
            "Top-K results",
            min_value=3,
            max_value=15,
            value=int(st.session_state.config['top_k']),
            help="Number of similar frames to show"
        )
        
        # Model selection dropdown
        model_name = st.selectbox(
            "CLIP Model",
            options=["openai/clip-vit-base-patch32", "openai/clip-vit-large-patch14"],
            index=0,
            help="CLIP model variant (base is faster, large is more accurate)"
        )
    
    st.markdown("### 🔍 Query Frames")
    st.markdown("Select which frames to use as queries (0 = first frame)")
    
    # Convert query indices to string for display
    current_queries = st.session_state.config.get('query_indices', [0, 15, 25])
    query_str = ", ".join(str(x) for x in current_queries)
    
    query_input = st.text_input(
        "Frame indices (comma-separated)",
        value=query_str,
        help="Example: 0, 15, 25 for frames 0, 15, and 25"
    )
    
    try:
        query_indices = [int(x.strip()) for x in query_input.split(",") if x.strip()]
    except:
        st.error("Invalid format. Use numbers separated by commas.")
        query_indices = [0, 15, 25]
    
    if st.button("💾 Save Configuration", type="primary", use_container_width=True):
        # Update all config keys
        st.session_state.config.update({
            'video_folder': 'videos',
            'frames_folder': 'frames',
            'frames_per_video': frames_per_video,
            'frame_rate': float(frame_rate),
            'model_name': model_name,
            'batch_size': batch_size,
            'embedding_dim': 512,
            'top_k': top_k,
            'query_indices': query_indices
        })
        
        # Save to config.yaml with all keys
        with open('config.yaml', 'w') as f:
            yaml.dump(st.session_state.config, f, default_flow_style=False)
        
        st.success("✅ Configuration saved!")
        
        # Show saved config
        with st.expander("View Saved Configuration"):
            st.json(st.session_state.config)

elif st.session_state.page == "Run":
    st.markdown('<h1 class="main-header">🚀 Run Pipeline</h1>', unsafe_allow_html=True)
    
    # Check if videos exist
    video_count = len(list(Path('videos').glob('*.mp4')))
    
    if video_count < 2:
        st.warning(f"⚠️ Need at least 2 videos. Currently have {video_count}")
        if st.button("📤 Go to Upload Page"):
            st.session_state.page = "Upload"
            st.rerun()
    else:
        st.success(f"✅ {video_count} videos ready")
        
        # Show video list
        with st.expander("📋 Videos to process"):
            for v in list(Path('videos').glob('*.mp4')):
                file_size = v.stat().st_size / (1024 * 1024)
                st.text(f"📹 {v.name} ({file_size:.1f} MB)")
        
        # Show current config
        st.markdown("### ⚙️ Current Configuration")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Frames/Video", st.session_state.config['frames_per_video'])
        with col2:
            st.metric("Frame Rate", f"{st.session_state.config['frame_rate']} fps")
        with col3:
            st.metric("Batch Size", st.session_state.config['batch_size'])
        with col4:
            st.metric("Top-K", st.session_state.config['top_k'])
        
        # Show model info
        st.info(f"**Model:** {st.session_state.config.get('model_name', 'Not set')}")
        
        # Run button
        if st.button("▶️ START PIPELINE", type="primary", use_container_width=True):
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            try:
                # Step 1: Clear previous outputs
                status_text.text("Step 1/5: Cleaning previous outputs...")
                if Path('frames').exists():
                    shutil.rmtree('frames')
                if Path('embeddings.npy').exists():
                    Path('embeddings.npy').unlink()
                if Path('metadata.csv').exists():
                    Path('metadata.csv').unlink()
                if Path('similarity_heatmap.png').exists():
                    Path('similarity_heatmap.png').unlink()
                
                Path('frames').mkdir(exist_ok=True)
                progress_bar.progress(10)
                
                # Step 2: Save config with ALL required keys
                status_text.text("Step 2/5: Saving configuration...")
                
                # Ensure config has all required keys with correct types
                full_config = {
                    'video_folder': st.session_state.config.get('video_folder', 'videos'),
                    'frames_folder': st.session_state.config.get('frames_folder', 'frames'),
                    'frames_per_video': int(st.session_state.config.get('frames_per_video', 30)),
                    'frame_rate': float(st.session_state.config.get('frame_rate', 1.0)),
                    'model_name': st.session_state.config.get('model_name', 'openai/clip-vit-base-patch32'),
                    'batch_size': int(st.session_state.config.get('batch_size', 16)),
                    'embedding_dim': 512,
                    'top_k': int(st.session_state.config.get('top_k', 5)),
                    'query_indices': st.session_state.config.get('query_indices', [0, 15, 25])
                }
                
                with open('config.yaml', 'w') as f:
                    yaml.dump(full_config, f, default_flow_style=False)
                    
                progress_bar.progress(20)
                
                # Step 3: Run the pipeline
                status_text.text("Step 3/5: Running pipeline... (this may take a few minutes)")
                progress_bar.progress(30)
                
                # Execute run_pipeline.py
                result = subprocess.run(
                    [sys.executable, 'run_pipeline.py'],
                    capture_output=True,
                    text=True
                )
                
                # Show output in expander
                with st.expander("📋 Pipeline Output"):
                    st.text(result.stdout)
                    if result.stderr:
                        st.error(result.stderr)
                
                if result.returncode != 0:
                    st.error("❌ Pipeline failed!")
                    if result.stderr:
                        st.code(result.stderr)
                        
                    st.info("""
                    **Troubleshooting:**
                    - Check that your videos are valid MP4 files
                    - Make sure you have at least 2 videos
                    - Check config.yaml has all required keys
                    - Try running 'python run_pipeline.py' directly in terminal
                    """)
                else:
                    progress_bar.progress(70)
                    status_text.text("Step 4/5: Loading results...")
                    
                    # Verify outputs
                    if Path('embeddings.npy').exists():
                        embeddings = np.load('embeddings.npy')
                        st.success(f"✅ Generated {len(embeddings)} embeddings")
                    
                    if Path('metadata.csv').exists():
                        metadata = pd.read_csv('metadata.csv')
                        st.success(f"✅ Metadata saved for {len(metadata)} frames")
                    
                    if Path('similarity_heatmap.png').exists():
                        st.success("✅ Similarity heatmap created")
                    
                    progress_bar.progress(90)
                    status_text.text("Step 5/5: Finalizing...")
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Pipeline complete!")
                    st.balloons()
                    
                    st.session_state.pipeline_complete = True
                    
                    time.sleep(2)
                    progress_bar.empty()
                    status_text.empty()
                    
            except Exception as e:
                st.error(f"Error: {e}")
                import traceback
                st.code(traceback.format_exc())
        
        # Results preview if complete
        if st.session_state.pipeline_complete:
            st.markdown("### 📊 Quick Preview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                if Path('metadata.csv').exists():
                    metadata = pd.read_csv('metadata.csv')
                    st.dataframe(metadata.head(), use_container_width=True)
            
            with col2:
                if Path('similarity_heatmap.png').exists():
                    st.image('similarity_heatmap.png', caption="Similarity Heatmap", use_container_width=True)

elif st.session_state.page == "Explore":
    st.markdown('<h1 class="main-header">🔍 Explore Results with AI Insights</h1>', unsafe_allow_html=True)
    
    # Check if outputs exist
    if not Path('embeddings.npy').exists() or not Path('metadata.csv').exists():
        st.warning("⚠️ No results found. Please run the pipeline first.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🚀 Go to Run Page"):
                st.session_state.page = "Run"
                st.rerun()
        with col2:
            if st.button("📤 Upload Videos"):
                st.session_state.page = "Upload"
                st.rerun()
    else:
        # Load data
        embeddings = np.load('embeddings.npy')
        metadata = pd.read_csv('metadata.csv')
        
        st.success(f"✅ Loaded {len(embeddings)} embeddings and {len(metadata)} frames")
        
        # Get list of frame files
        frame_files = sorted(Path('frames').glob('*.jpg'))
        
        if len(frame_files) == 0:
            st.error("No frame files found in frames/ directory")
        else:
            # Query selection
            st.markdown("### 🖼️ Select a Query Frame")
            st.markdown(f"Total frames available: {len(frame_files)}")
            
            # Create two columns for selection
            col1, col2 = st.columns([1, 1])
            
            with col1:
                # Add a dropdown for quick navigation
                frame_indices = list(range(len(frame_files)))
                selected_idx_dropdown = st.selectbox(
                    "Jump to frame:",
                    frame_indices,
                    format_func=lambda x: f"Frame {x} - {frame_files[x].name}"
                )
            
            with col2:
                # Quick stats about the selected frame
                if selected_idx_dropdown < len(metadata):
                    st.info(f"📊 Frame {selected_idx_dropdown} is from {metadata.iloc[selected_idx_dropdown]['video_id']} at {metadata.iloc[selected_idx_dropdown]['timestamp']:.1f}s")
            
            # Display frame grid (show first 20 frames)
            st.markdown("### 🖼️ Frame Gallery")
            st.markdown("Click any frame to see similar ones with AI explanations:")
            
            cols_per_row = 4
            selected_idx = None
            
            # Create a grid of frames
            for i in range(0, min(len(frame_files), 20), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    idx = i + j
                    if idx < len(frame_files) and idx < 20:
                        with col:
                            # Display frame
                            st.image(str(frame_files[idx]), use_container_width=True)
                            st.caption(f"Frame {idx}")
                            
                            # Select button
                            if st.button(f"🔍 Select #{idx}", key=f"select_{idx}", use_container_width=True):
                                selected_idx = idx
                                st.session_state.selected_query = idx
            
            # Also allow selection from dropdown
            if st.button(f"🔍 Select Frame #{selected_idx_dropdown}", key="select_dropdown", type="primary", use_container_width=True):
                selected_idx = selected_idx_dropdown
                st.session_state.selected_query = selected_idx_dropdown
            
            # Use session state to remember selection
            if st.session_state.selected_query is not None:
                selected_idx = st.session_state.selected_query
            
            # Show results if a frame was selected
            if selected_idx is not None and selected_idx < len(frame_files):
                st.markdown("---")
                st.markdown(f"### 🎯 Results for Frame {selected_idx}")
                
                # Compute similarities
                query_embedding = embeddings[selected_idx].reshape(1, -1)
                similarities = cosine_similarity(query_embedding, embeddings)[0]
                
                # Get top indices (excluding self)
                top_k = st.session_state.config.get('top_k', 5)
                top_indices = np.argsort(similarities)[::-1][1:top_k+1]
                
                # Prepare data for AI explanation
                similar_frames = [frame_files[idx].name for idx in top_indices]
                similar_scores = [similarities[idx] for idx in top_indices]
                
                # Show AI explanation if enabled
                if enable_ai:
                    with st.spinner("🤖 AI is analyzing why these frames are similar..."):
                        explanation = get_ai_explanation(
                            frame_files[selected_idx].name,
                            similar_frames,
                            similar_scores
                        )
                        
                        st.markdown(f"""
                        <div class="explanation-box">
                            <h4>🤖 AI Insight</h4>
                            {explanation}
                        </div>
                        """, unsafe_allow_html=True)
                
                # Display query and similar frames side by side
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("**🔍 Query Frame:**")
                    st.image(str(frame_files[selected_idx]), use_container_width=True)
                    
                    # Show metadata
                    if selected_idx < len(metadata):
                        st.caption(f"📹 Video: {metadata.iloc[selected_idx]['video_id']}")
                        st.caption(f"⏱️ Timestamp: {metadata.iloc[selected_idx]['timestamp']:.1f}s")
                        st.caption(f"📄 File: {metadata.iloc[selected_idx]['filename']}")
                
                with col2:
                    st.markdown("**🎯 Similar Frames:**")
                    
                    for i, idx in enumerate(top_indices):
                        with st.container():
                            cols = st.columns([1, 2, 1])
                            
                            with cols[0]:
                                st.markdown(f"**#{i+1}**")
                            
                            with cols[1]:
                                if idx < len(frame_files):
                                    st.image(str(frame_files[idx]), width=200)
                            
                            with cols[2]:
                                score = similarities[idx]
                                if score > 0.8:
                                    st.markdown(f'<span class="score-high">✨ Score: {score:.3f}</span>', 
                                              unsafe_allow_html=True)
                                    st.caption("Very similar vibe")
                                elif score > 0.6:
                                    st.markdown(f'<span class="score-medium">📊 Score: {score:.3f}</span>', 
                                              unsafe_allow_html=True)
                                    st.caption("Moderately similar")
                                else:
                                    st.markdown(f'<span class="score-low">📉 Score: {score:.3f}</span>', 
                                              unsafe_allow_html=True)
                                    st.caption("Somewhat similar")
                                
                                if idx < len(metadata):
                                    st.caption(f"📹 {metadata.iloc[idx]['video_id']}")
                            
                            st.markdown("---")
        
        # Show heatmap with interpretation
        st.markdown("### 📊 Similarity Heatmap")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if Path('similarity_heatmap.png').exists():
                st.image('similarity_heatmap.png', use_container_width=True)
            else:
                # Generate on the fly
                with st.spinner("Generating heatmap..."):
                    sim_matrix = cosine_similarity(embeddings)
                    
                    fig, ax = plt.subplots(figsize=(10, 8))
                    im = ax.imshow(sim_matrix, cmap='viridis', vmin=-1, vmax=1)
                    plt.colorbar(im, ax=ax)
                    plt.title('Frame Similarity Matrix')
                    plt.xlabel('Frame Index')
                    plt.ylabel('Frame Index')
                    st.pyplot(fig)
        
        with col2:
            st.markdown("**📖 How to read this:**")
            st.markdown("""
            - **🟡 Yellow squares**: Frames that look very similar
            - **🟣 Purple squares**: Frames that look different
            - **📈 Diagonal line**: Each frame compared to itself (perfect match)
            - **🔴 Red lines**: Where one video ends and another begins
            
            **What to look for:**
            - Yellow blocks outside the diagonal mean different videos have similar moments
            - The brighter the yellow, the stronger the visual connection
            """)
            
            # Add AI summary of the heatmap
            if enable_ai and Path('similarity_heatmap.png').exists():
                st.markdown("**🤖 Quick AI Summary:**")
                
                # Calculate some simple stats
                sim_matrix = cosine_similarity(embeddings)
                avg_similarity = np.mean(sim_matrix[sim_matrix < 0.99])  # Exclude self
                
                if avg_similarity > 0.7:
                    st.success(f"✨ All your videos share a consistent visual style (average similarity: {avg_similarity:.2f})")
                elif avg_similarity > 0.5:
                    st.info(f"📊 Your videos have some visual connections (average similarity: {avg_similarity:.2f})")
                else:
                    st.warning(f"🎨 Your videos are quite visually distinct (average similarity: {avg_similarity:.2f})")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        GenTA GACS Mini - Affective Computing for Art & Marketing<br>
        Upload → Configure → Run → Explore with AI Insights
    </div>
    """,
    unsafe_allow_html=True
)
