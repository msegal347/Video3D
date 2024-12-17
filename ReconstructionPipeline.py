import argparse
from pathlib import Path
import subprocess
import ffmpeg
from feature_extractor import FeatureExtractor
from feature_matcher import FeatureMatcher
from sparse_reconstruction import SparseReconstruction
from dense_reconstruction import DenseReconstruction
from visualization import ReconstructionVisualizer
from utils.frame_extraction import extract_frames


class Config:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.image_dir = self.project_dir / "images"
        self.database_path = self.project_dir / "database.db"
        self.sparse_dir = self.project_dir / "sparse"
        self.dense_dir = self.project_dir / "dense"

    def create_directories(self):
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)


class UnifiedPipeline:
    def __init__(self, video_path, project_dir, colmap_path):
        self.video_path = Path(video_path).resolve()
        self.project_dir = Path(project_dir).resolve()
        self.config = Config(self.project_dir)
        self.colmap_path = colmap_path

    def extract_frames(self, fps=1):
        """Extract frames from the video using FFmpeg."""
        print(f"Extracting frames from {self.video_path} to {self.config.image_dir}")
        self.config.create_directories()
        output_pattern = self.config.image_dir / "frame_%04d.png"

        (
            ffmpeg
            .input(str(self.video_path))
            .output(str(output_pattern), vf=f"fps={fps}")
            .run(overwrite_output=True)
        )
        print(f"Frames extracted to: {self.config.image_dir}")

    def extract_features(self):
        """Extract SIFT features."""
        print("Extracting features...")
        extractor = FeatureExtractor(self.config)
        extractor.extract_features()

    def match_features(self):
        """Match features using Exhaustive Matching."""
        print("Matching features...")
        matcher = FeatureMatcher(self.config)
        matcher.match_features()

    def sparse_reconstruction(self):
        """Perform incremental sparse reconstruction."""
        print("Running sparse reconstruction...")
        reconstructor = SparseReconstruction(self.config)
        reconstructor.incremental_mapping()

    def dense_reconstruction(self):
        """Perform dense reconstruction."""
        print("Running dense reconstruction...")
        reconstructor = DenseReconstruction(self.config, self.colmap_path)
        reconstructor.undistort_images()
        reconstructor.patch_match_stereo()
        reconstructor.stereo_fusion()

    def visualize_reconstruction(self, reconstruction_type):
        """Visualize sparse or dense reconstruction."""
        print(f"Visualizing {reconstruction_type} reconstruction...")
        visualizer = ReconstructionVisualizer(self.project_dir, self.colmap_path)
        if reconstruction_type == "sparse":
            visualizer.visualize_sparse()
        elif reconstruction_type == "dense":
            visualizer.visualize_dense()

    def run(self, fps=1, visualize=False):
        """Run the entire pipeline."""
        print("Starting unified pipeline...")
        self.extract_frames(fps=fps)
        self.extract_features()
        self.match_features()
        self.sparse_reconstruction()
        self.dense_reconstruction()

        if visualize:
            self.visualize_reconstruction("sparse")
            self.visualize_reconstruction("dense")
        print("Pipeline completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified 3D Reconstruction Pipeline")
    parser.add_argument(
        "--video_path", type=str, required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--project_dir", type=str, required=True,
        help="Path to the output project directory"
    )
    parser.add_argument(
        "--colmap_path", type=str, required=True,
        help="Path to the COLMAP executable or COLMAP.bat"
    )
    parser.add_argument(
        "--fps", type=int, default=1,
        help="Frames per second to extract from the video (default: 1)"
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Visualize the sparse and dense reconstructions"
    )
    args = parser.parse_args()

    # Initialize and run the pipeline
    pipeline = UnifiedPipeline(args.video_path, args.project_dir, args.colmap_path)
    pipeline.run(fps=args.fps, visualize=args.visualize)
