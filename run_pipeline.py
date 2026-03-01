"""
GenTA GACS Mini Pipeline - FINAL VERSION (No Unicode Issues)
Main script that orchestrates the entire affective computing pipeline.
"""

import os
import cv2
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import yaml
from tqdm import tqdm
from PIL import Image
import argparse
import time
import warnings
warnings.filterwarnings('ignore')

# Suppress HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Fix for Windows console encoding
import sys
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import (
    logger, verify_embeddings, validate_video, 
    save_metadata, get_video_duration
)


class VideoProcessor:
    """Handles video loading and frame extraction."""
    
    def __init__(self, config: dict):
        """
        Initialize video processor with configuration.
        
        Args:
            config: Dictionary containing pipeline configuration
        """
        self.config = config
        self.frames_folder = Path(config['frames_folder'])
        self.frames_folder.mkdir(exist_ok=True)
    
    def extract_frames(self, video_path: Path, video_id: str) -> list:
        """
        Extract frames from a single video at specified rate.
        
        Args:
            video_path: Path to video file
            video_id: Identifier for the video
        
        Returns:
            List of frame metadata dictionaries
        """
        frames_data = []
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            logger.error(f"Cannot open {video_path}")
            return frames_data
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        logger.info(f"Extracting from {video_path.name} ({fps:.2f} fps, {duration:.1f}s)")
        
        # Calculate sampling interval (1 frame per second)
        interval = int(fps / self.config['frame_rate'])
        if interval < 1:
            interval = 1
        
        frame_count = 0
        saved_count = 0
        max_frames = self.config['frames_per_video']
        
        while saved_count < max_frames and frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Save frame at specified interval
            if frame_count % interval == 0:
                timestamp = frame_count / fps
                filename = f"{video_id}_frame_{saved_count:03d}_t{timestamp:.1f}s.jpg"
                filepath = self.frames_folder / filename
                
                # OPTIMIZATION: Resize frame to CLIP input size (224x224)
                frame_resized = cv2.resize(frame, (224, 224))
                
                # Save frame
                cv2.imwrite(str(filepath), frame_resized, [cv2.IMWRITE_JPEG_QUALITY, 85])
                
                # Verify save and add to metadata
                if filepath.exists():
                    frames_data.append({
                        'video_id': video_id,
                        'video_name': video_path.stem,
                        'frame_id': f"{video_id}_{saved_count:03d}",
                        'frame_index': saved_count,
                        'timestamp': timestamp,
                        'filepath': str(filepath),
                        'filename': filename
                    })
                    saved_count += 1
                    
                    # Log progress periodically
                    if saved_count % 5 == 0:
                        logger.info(f"    Saved {saved_count}/{max_frames} frames")
            
            frame_count += 1
        
        cap.release()
        logger.info(f"  -> Completed: {len(frames_data)} frames from {video_id}")
        return frames_data
    
    def extract_all_frames(self, video_files: list) -> tuple:
        """
        Extract frames from all videos.
        
        Args:
            video_files: List of paths to video files
        
        Returns:
            Tuple of (frame_paths list, metadata list)
        """
        all_frame_paths = []
        all_metadata = []
        
        start_time = time.time()
        
        for i, video_path in enumerate(video_files):
            video_id = f"video_{i+1}"
            logger.info(f"\nProcessing video {i+1}/{len(video_files)}: {video_path.name}")
            frames = self.extract_frames(video_path, video_id)
            
            for frame in frames:
                all_frame_paths.append(frame['filepath'])
                all_metadata.append(frame)
        
        elapsed = time.time() - start_time
        logger.info(f"\n✓ Total frames extracted: {len(all_frame_paths)} in {elapsed:.1f}s")
        return all_frame_paths, all_metadata


class EmbeddingModel:
    """CLIP model wrapper with verification and caching."""
    
    _model_cache = None
    _processor_cache = None
    
    def __init__(self, config: dict):
        """
        Initialize CLIP model.
        
        Args:
            config: Dictionary containing model configuration
        """
        self.config = config
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if EmbeddingModel._model_cache is not None:
            self.model = EmbeddingModel._model_cache
            self.processor = EmbeddingModel._processor_cache
            logger.info(f"Using cached CLIP model on {self.device}")
        else:
            logger.info(f"Loading CLIP model on {self.device}...")
            start_time = time.time()
            
            # Load model and processor
            self.model = CLIPModel.from_pretrained(config['model_name'])
            self.processor = CLIPProcessor.from_pretrained(config['model_name'])
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Cache for next time
            EmbeddingModel._model_cache = self.model
            EmbeddingModel._processor_cache = self.processor
            
            elapsed = time.time() - start_time
            logger.info(f"Model loaded in {elapsed:.1f}s")
    
    def _get_image_features(self, images):
        """
        Helper method to get image features correctly.
        
        Args:
            images: List of PIL Images or single PIL Image
        
        Returns:
            Normalized image features tensor
        """
        inputs = self.processor(images=images, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        with torch.no_grad():
            # Get vision model outputs
            vision_outputs = self.model.vision_model(pixel_values=pixel_values)
            
            # Extract pooled output (CLS token)
            pooled_output = vision_outputs.pooler_output
            
            # Apply projection to get image features
            image_features = self.model.visual_projection(pooled_output)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features
    
    def compute_embeddings(self, image_paths: list) -> np.ndarray:
        """
        Compute embeddings for multiple images in batches.
        
        Args:
            image_paths: List of paths to image files
        
        Returns:
            Numpy array of embeddings (N x embedding_dim)
        """
        embeddings = []
        batch_size = self.config['batch_size']
        
        logger.info(f"Computing embeddings for {len(image_paths)} images...")
        pbar = tqdm(total=len(image_paths), desc="Computing embeddings")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i + batch_size]
            batch_images = []
            
            # Load batch images
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_images.append(img)
                except Exception as e:
                    logger.error(f"Failed to load {path}: {e}")
                    # Add a black image as placeholder
                    batch_images.append(Image.new('RGB', (224, 224), color='black'))
            
            if batch_images:
                try:
                    # Get features using our helper method
                    batch_features = self._get_image_features(batch_images)
                    embeddings.append(batch_features.cpu().numpy())
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
            
            pbar.update(len(batch_paths))
        
        pbar.close()
        
        if not embeddings:
            logger.error("No embeddings computed!")
            return np.array([])
        
        try:
            all_embeddings = np.vstack(embeddings)
            logger.info(f"✓ Successfully computed {len(all_embeddings)} embeddings of dimension {all_embeddings.shape[1]}")
            
            # Log sample stats
            logger.info(f"  Sample embedding stats - min: {np.min(all_embeddings[0]):.3f}, "
                       f"max: {np.max(all_embeddings[0]):.3f}, "
                       f"mean: {np.mean(all_embeddings[0]):.3f}")
            
            return all_embeddings
        except Exception as e:
            logger.error(f"Failed to combine embeddings: {e}")
            return np.array([])


class SimilarityAnalyzer:
    """Computes and visualizes similarities between frames."""
    
    def __init__(self, embeddings: np.ndarray, metadata: list):
        self.embeddings = embeddings
        self.metadata = metadata
        self.similarity_matrix = None
    
    def compute_similarity(self) -> np.ndarray:
        """Compute pairwise cosine similarity matrix."""
        if len(self.embeddings) == 0:
            logger.error("No embeddings to compute similarity!")
            return np.array([])
        
        logger.info(f"Computing similarity matrix for {len(self.embeddings)} frames...")
        start_time = time.time()
        
        self.similarity_matrix = cosine_similarity(self.embeddings)
        self.similarity_matrix = np.clip(self.similarity_matrix, -1, 1)
        
        elapsed = time.time() - start_time
        logger.info(f"✓ Computed similarity matrix: {self.similarity_matrix.shape} in {elapsed:.1f}s")
        logger.info(f"  Similarity range: [{np.min(self.similarity_matrix):.3f}, {np.max(self.similarity_matrix):.3f}]")
        
        return self.similarity_matrix
    
    def find_top_k_similar(self, query_idx: int, k: int = 5) -> list:
        """Find top k most similar frames to query frame."""
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        if len(self.similarity_matrix) == 0:
            return []
        
        similarities = self.similarity_matrix[query_idx]
        sorted_indices = np.argsort(similarities)[::-1]
        top_indices = [idx for idx in sorted_indices if idx != query_idx][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                'rank': len(results) + 1,
                'frame': self.metadata[idx]['filename'],
                'filepath': self.metadata[idx]['filepath'],
                'video': self.metadata[idx]['video_id'],
                'timestamp': self.metadata[idx]['timestamp'],
                'score': float(similarities[idx])
            })
        
        return results
    
    def print_query_results(self, query_idx: int, k: int = 5):
        """Print query results in a readable format."""
        if query_idx >= len(self.metadata):
            logger.warning(f"Query index {query_idx} out of range")
            return
        
        results = self.find_top_k_similar(query_idx, k)
        query_frame = self.metadata[query_idx]
        
        # FIXED: Removed all emojis, using ASCII only
        print("\n" + "=" * 70)
        print(f"QUERY FRAME [{query_idx}]: {query_frame['filename']}")
        print(f"   Video: {query_frame['video_id']}, Timestamp: {query_frame['timestamp']:.1f}s")
        print("=" * 70)
        print("TOP SIMILAR FRAMES:")
        
        for r in results:
            similarity_bar = "*" * int(r['score'] * 20)
            print(f"\n  #{r['rank']} | Score: {r['score']:.3f} {similarity_bar}")
            print(f"     Frame: {r['frame']}")
            print(f"     Video: {r['video']}, Timestamp: {r['timestamp']:.1f}s")
        
        print("\n" + "=" * 70)
    
    def plot_heatmap(self, output_path: str = "similarity_heatmap.png"):
        """Create and save similarity heatmap."""
        if self.similarity_matrix is None:
            self.compute_similarity()
        
        if len(self.similarity_matrix) == 0:
            logger.error("Cannot plot heatmap: no similarity matrix")
            return
        
        logger.info("Generating heatmap...")
        plt.figure(figsize=(14, 12))
        
        # Create heatmap
        plt.imshow(self.similarity_matrix, cmap='viridis', vmin=-1, vmax=1, aspect='auto')
        plt.colorbar(label='Cosine Similarity', shrink=0.8)
        plt.title('Frame Similarity Matrix', fontsize=16, pad=20)
        plt.xlabel('Frame Index', fontsize=12)
        plt.ylabel('Frame Index', fontsize=12)
        
        # Add video boundaries
        if len(self.metadata) > 0:
            video_changes = []
            current_video = self.metadata[0]['video_id']
            for i, meta in enumerate(self.metadata):
                if meta['video_id'] != current_video:
                    video_changes.append(i)
                    current_video = meta['video_id']
            
            for change in video_changes:
                plt.axhline(y=change, color='red', linestyle='--', alpha=0.5, linewidth=1)
                plt.axvline(x=change, color='red', linestyle='--', alpha=0.5, linewidth=1)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"✓ Heatmap saved to {output_path}")


def validate_config(config: dict) -> bool:
    """Validate that config has all required keys."""
    required_keys = [
        'video_folder', 'frames_folder', 'frames_per_video',
        'frame_rate', 'model_name', 'batch_size', 
        'embedding_dim', 'query_indices', 'top_k'
    ]
    
    missing_keys = [key for key in required_keys if key not in config]
    
    if missing_keys:
        logger.error(f"Missing required config keys: {missing_keys}")
        return False
    
    return True


def main():
    """Run the complete GenTA GACS Mini pipeline."""
    
    # Print header
    logger.info("=" * 70)
    logger.info("GenTA GACS Mini Pipeline")
    logger.info("=" * 70)
    
    # Load configuration
    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning("config.yaml not found! Creating default...")
        default_config = {
            'video_folder': 'videos',
            'frames_folder': 'frames',
            'frames_per_video': 30,
            'frame_rate': 1.0,
            'model_name': 'openai/clip-vit-base-patch32',
            'batch_size': 16,
            'embedding_dim': 512,
            'query_indices': [0, 15, 25],
            'top_k': 5
        }
        
        with open('config.yaml', 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        logger.info("✓ Default config.yaml created")
        config = default_config
    else:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        logger.info("✓ Configuration loaded successfully")
    
    # Validate config
    if not validate_config(config):
        logger.error("Invalid configuration!")
        return 1
    
    logger.info("Configuration:")
    logger.info(f"  - Videos folder: {config['video_folder']}")
    logger.info(f"  - Frames per video: {config['frames_per_video']}")
    logger.info(f"  - Frame rate: {config['frame_rate']} fps")
    logger.info(f"  - Model: {config['model_name']}")
    logger.info(f"  - Batch size: {config['batch_size']}")
    
    # Find videos
    video_folder = Path(config['video_folder'])
    if not video_folder.exists():
        video_folder.mkdir(exist_ok=True)
        logger.warning(f"Created {video_folder} folder. Please add MP4 videos.")
        return 1
    
    video_files = list(video_folder.glob('*.mp4')) + list(video_folder.glob('*.MP4'))
    
    if len(video_files) < 2:
        logger.error(f"❌ Need at least 2 videos. Found {len(video_files)}")
        return 1
    
    logger.info(f"✓ Found {len(video_files)} videos")
    for v in video_files:
        logger.info(f"    {v.name}")
    
    # Validate videos
    logger.info("\nValidating videos...")
    valid_videos = []
    for v in video_files:
        if validate_video(v):
            valid_videos.append(v)
            logger.info(f"    ✓ {v.name}")
    
    if len(valid_videos) < 2:
        logger.error(f"❌ Only {len(valid_videos)} valid videos.")
        return 1
    
    # Step 1: Extract frames
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: Extracting frames")
    logger.info("=" * 70)
    
    processor = VideoProcessor(config)
    frame_paths, metadata = processor.extract_all_frames(valid_videos)
    
    if len(frame_paths) < 10:
        logger.error(f"❌ Only extracted {len(frame_paths)} frames.")
        return 1
    
    logger.info(f"\n✓ Successfully extracted {len(frame_paths)} frames to '{config['frames_folder']}/'")
    
    # Step 2: Compute embeddings
    logger.info("\n" + "=" * 70)
    logger.info("STEP 2: Computing CLIP embeddings")
    logger.info("=" * 70)
    
    model = EmbeddingModel(config)
    embeddings = model.compute_embeddings(frame_paths)
    
    if len(embeddings) == 0:
        logger.error("❌ No embeddings computed!")
        return 1
    
    # Save embeddings and metadata
    np.save('embeddings.npy', embeddings)
    save_metadata(metadata, 'metadata.csv')
    logger.info("✓ Saved embeddings to 'embeddings.npy'")
    logger.info("✓ Saved metadata to 'metadata.csv'")
    
    # Step 3: Similarity analysis
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: Analyzing similarities")
    logger.info("=" * 70)
    
    analyzer = SimilarityAnalyzer(embeddings, metadata)
    similarity_matrix = analyzer.compute_similarity()
    
    if len(similarity_matrix) == 0:
        logger.error("❌ No similarity matrix computed!")
        return 1
    
    # Step 4: Find and display similar frames for queries
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: Finding similar frames")
    logger.info("=" * 70)
    
    for query_idx in config['query_indices']:
        analyzer.print_query_results(query_idx, config['top_k'])
    
    # Step 5: Create visualization
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: Creating visualization")
    logger.info("=" * 70)
    
    analyzer.plot_heatmap('similarity_heatmap.png')
    
    # Final summary
    logger.info("\n" + "=" * 70)
    logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    logger.info("=" * 70)
    logger.info("\nOUTPUT FILES GENERATED:")
    logger.info(f"  📁 '{config['frames_folder']}/' - {len(frame_paths)} extracted frames")
    logger.info(f"  📄 'embeddings.npy' - {embeddings.shape[0]} embeddings of dim {embeddings.shape[1]}")
    logger.info(f"  📄 'metadata.csv' - Frame information")
    logger.info(f"  🖼️  'similarity_heatmap.png' - Similarity visualization")
    
    # Print sample of results
    logger.info("\nSAMPLE RESULTS (first query):")
    if len(config['query_indices']) > 0 and config['query_indices'][0] < len(metadata):
        first_query = config['query_indices'][0]
        results = analyzer.find_top_k_similar(first_query, 3)
        logger.info(f"  Query frame: {metadata[first_query]['filename']}")
        for r in results:
            logger.info(f"    -> {r['frame']} (score: {r['score']:.3f})")
    
    logger.info("\n" + "=" * 70)
    
    return 0


if __name__ == "__main__":
    exit(main())