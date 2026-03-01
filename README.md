# 🎨 GenTA GACS Mini: Affective Computing for Art & Marketing Videos

## 📋 **Quick Overview**

**GenTA GACS Mini** is a prototype affective computing pipeline that analyzes the "feel" or "vibe" of art and marketing videos. It extracts frames, computes CLIP embeddings, and finds visually similar frames across different videos—transforming subjective emotional responses into measurable data.

> *"Contemporary art and marketing visuals evoke emotional responses that are difficult to quantify. Traditional metadata fails to capture the actual 'feel' of visual content. This tool makes art more emotionally accessible by understanding content based on affective qualities."*

---

## ✨ **What This Tool Does**

| Step | Description |
|------|-------------|
| **1. Frame Extraction** | Takes snapshots from your videos (1 frame per second) |
| **2. AI Analysis** | Converts each frame into a 512-number "fingerprint" (embedding) using CLIP |
| **3. Similarity Matching** | Compares fingerprints to find frames with similar visual "feel" |
| **4. Interactive Exploration** | Lets you click any frame and instantly find others with the same vibe |

---

## 🚀 **5-Minute Quick Start**

### **Prerequisites**
- ✅ Python 3.10 installed ([download](https://www.python.org/downloads/))
- ✅ 2-3 short MP4 videos (art, abstract, or marketing)
- ✅ 4GB RAM minimum (8GB recommended)

### **Installation**

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/genta-gacs-mini.git
cd genta-gacs-mini

# 2. Create virtual environment with Python 3.10
py -3.10 -m venv venv

# 3. Activate it (Windows)
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Add your videos
# Create a 'videos' folder and drop 2-3 MP4 files inside
mkdir videos

# 6. Launch the app!
streamlit run app.py
Your browser will automatically open to http://localhost:8501

🎯 Understanding the Results
What the Similarity Scores Mean
Score Range	Meaning	Example
1.0	Perfect match - visually identical	Same scene from same video
0.8 - 0.99	Very similar vibe	Same lighting, colors, composition
0.6 - 0.79	Moderately similar	Shared style elements
< 0.6	Different vibe	Distinct visual languages
Example Output
text
QUERY FRAME [0]: video_1_frame_000_t0.0s.jpg
TOP SIMILAR FRAMES:
  #1 | Score: 1.000  → video_3_frame_000_t0.0s.jpg (identical vibe)
  #2 | Score: 0.926  → video_3_frame_007_t7.0s.jpg (very similar)
  #3 | Score: 0.926  → video_1_frame_007_t7.0s.jpg (same video, similar moment)
📖 User Guide: Navigating the App
Sidebar Navigation
text
🏠 Home           → Overview and quick status
📤 Upload Videos  → Add your MP4 files
⚙️ Configure      → Adjust pipeline settings
🚀 Run Pipeline   → Process videos and generate embeddings
🔍 Explore Results → Interactively find similar frames
1. 🏠 Home Page
Check your current status at a glance. The right panel shows:

How many videos you've uploaded

How many frames have been extracted

Whether embeddings are ready

2. 📤 Upload Videos Page
Step-by-step:

Click "Browse files" and select 2-3 MP4 videos

Click "💾 Save Videos" to upload

Verify they appear in the "Current Videos" list

*Requirements: MP4 format, 15-60 seconds each, art/abstract/marketing content*

3. ⚙️ Configure Page
Setting	What It Does	Recommended
Frames per video	Max snapshots to take	30
Frame rate (fps)	How many frames per second	1.0
Batch size	Images processed at once	16 (lower if slow)
Top-K results	How many similar frames to show	5
CLIP Model	Base (faster) or Large (more accurate)	Base for testing
Query indices	Which frames to use as examples	0, 15, 25
Always click "💾 Save Configuration" before moving on!

4. 🚀 Run Pipeline Page
This is where the magic happens. When you click "START PIPELINE":

text
Step 1/5: Cleaning previous outputs → Deletes old data
Step 2/5: Saving configuration → Writes your settings
Step 3/5: Running pipeline → AI processes your videos (2-5 minutes)
Step 4/5: Loading results → Prepares the data
Step 5/5: Finalizing → 🎉 Complete!
What's happening behind the scenes:

Videos are validated and frames extracted at 1 fps

Each frame is converted to a 512-dimensional CLIP embedding

Cosine similarity is calculated between all frames

Top-5 similar frames are identified for each query

A similarity heatmap is generated

5. 🔍 Explore Results Page
To find similar frames:

Browse the frame gallery or use the dropdown to jump to a specific frame

Click "Select" on any frame

Watch as the top-5 similar frames appear with:

Similarity scores (color-coded)

Visual previews

Video source and timestamp information

The heatmap at the bottom shows the full similarity matrix:

🟡 Yellow = high similarity

🟣 Purple = low similarity

🔴 Red lines = boundaries between videos

🧪 Verification Features
The pipeline includes built-in checks at every stage:

python
✓ Video validation (files exist, can be opened)
✓ Frame extraction verification (files saved correctly)
✓ NaN/Inf detection in embeddings
✓ Shape validation (512 dimensions)
✓ Identical image test (same image → same embedding)
✓ Similarity score range checking (-1 to 1)
🤖 AI Tools & Governance
This project leveraged AI coding assistants while maintaining human oversight:

Component	AI Tool	Human Verification
CLIP model integration	Copilot	Added error handling, tested with samples
Streamlit boilerplate	ChatGPT	Restructured, fixed type errors, added CSS
Frame extraction	Copilot	Added resize optimization, verified saving
Similarity calculation	Copilot	Added score clipping, validated ranges
All verification logic, error handling, and edge cases were human-written.

📁 Repository Structure
text
genta-gacs-mini/
├── README.md                 # This file
├── requirements.txt          # All dependencies
├── .gitignore               # Python + data exclusions
├── config.yaml              # Configuration template
├── run_pipeline.py          # Core pipeline (with verification)
├── app.py                   # Streamlit frontend
├── utils.py                 # Helper functions
├── tests/                   # Unit tests
│   ├── test_embeddings.py
│   └── test_frame_extractor.py
├── videos/                  # Place your MP4 files here
├── frames/                  # Extracted frames (auto-generated)
└── outputs/                 # Results (auto-generated)
    ├── embeddings.npy
    ├── metadata.csv
    └── similarity_heatmap.png
🔧 Troubleshooting
Problem	Solution
"No videos found"	Add MP4 files to videos/ folder
"Module not found"	Run pip install -r requirements.txt
"CUDA out of memory"	Set batch_size: 8 in config.yaml
Pipeline is slow	Reduce frames_per_video or use smaller videos
Unicode errors	Already fixed—run the latest code
Streamlit type errors	Restart the app with Ctrl+C then streamlit run app.py
🚀 Next Steps & Future Development
This prototype lays the groundwork for GenTA's full Affective Computing Engine:

Temporal analysis - Scene detection instead of uniform sampling

Multimodal integration - Add audio and text embeddings

Performance feedback - Connect vibe scores to CTR/ROAS

Affective dimensions - Map to specific moods (calm/exciting, warm/cool)

Scalable architecture - Vector database for fast similarity search

📄 License
MIT License - feel free to use, modify, and build upon this work.

👤 Author
Your Name - GenTA AI R&D Engineer Candidate

"Making art more emotionally accessible, one frame at a time."

🙏 Acknowledgments
OpenAI CLIP model via HuggingFace

Streamlit for the interactive dashboard

Pexels for royalty-free sample videos