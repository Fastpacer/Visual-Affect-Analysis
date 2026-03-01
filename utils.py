"""
Utility functions for GenTA GACS Mini pipeline.
Contains helper functions for logging, validation, and metadata management.
"""

import logging
import numpy as np
import cv2
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

def verify_embeddings(embeddings: np.ndarray, expected_dim: int = 512) -> bool:
    """
    Verify embeddings are valid (no NaN/Inf, correct shape).
    
    Args:
        embeddings: Numpy array of embeddings
        expected_dim: Expected embedding dimension
    
    Returns:
        True if valid, raises AssertionError otherwise
    """
    assert not np.isnan(embeddings).any(), "NaN values found in embeddings"
    assert not np.isinf(embeddings).any(), "Inf values found in embeddings"
    assert embeddings.ndim == 2, f"Expected 2D array, got {embeddings.ndim}D"
    assert embeddings.shape[1] == expected_dim, \
        f"Expected dim {expected_dim}, got {embeddings.shape[1]}"
    
    logger.info(f"✓ Embeddings verified: {embeddings.shape}")
    return True

def validate_video(video_path: Path) -> bool:
    """
    Check if video can be opened and has frames.
    
    Args:
        video_path: Path to video file
    
    Returns:
        True if valid, False otherwise
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return False
    
    # Try to read first frame
    ret, frame = cap.read()
    cap.release()
    
    if ret and frame is not None:
        logger.info(f"✓ Valid video: {video_path.name}")
        return True
    else:
        logger.error(f"Video has no readable frames: {video_path}")
        return False

def save_metadata(frames_data: List[Dict], output_path: str = "metadata.csv") -> str:
    """
    Save frame metadata to CSV.
    
    Args:
        frames_data: List of frame metadata dictionaries
        output_path: Path to save CSV file
    
    Returns:
        Path to saved file
    """
    df = pd.DataFrame(frames_data)
    df.to_csv(output_path, index=False)
    logger.info(f"✓ Saved metadata for {len(df)} frames to {output_path}")
    return output_path

def get_video_duration(video_path: Path) -> float:
    """
    Get video duration in seconds.
    
    Args:
        video_path: Path to video file
    
    Returns:
        Duration in seconds
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    
    if fps > 0:
        return frame_count / fps
    return 0.0