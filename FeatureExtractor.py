import pycolmap
import argparse
from pathlib import Path

class Config:
    def __init__(self, project_dir=None):
        # Set the project directory to the provided path or the current working directory
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.image_dir = self.project_dir / 'images'  # Directory with extracted frames
        self.database_path = self.project_dir / 'database.db'  # Feature database path
        self.sparse_dir = self.project_dir / 'sparse'  # Sparse reconstruction output
        self.dense_dir = self.project_dir / 'dense'  # Dense reconstruction output
        self.num_threads = 4
        self.max_image_size = 2400
        self.first_octave = -1

    def create_directories(self):
        # Ensure required directories exist, but do not overwrite existing content
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)

class FeatureExtractor:
    def __init__(self, config):
        self.config = config

    def extract_features(self):
        """
        Extract features using COLMAP, but skip extraction if database already exists.
        """
        if self.config.database_path.exists():
            print(f"Database already exists at: {self.config.database_path}. Skipping feature extraction.")
            return

        print("Extracting features...")
        sift_options = pycolmap.SiftExtractionOptions()
        sift_options.num_threads = self.config.num_threads
        sift_options.max_image_size = self.config.max_image_size
        sift_options.first_octave = self.config.first_octave

        # Run COLMAP feature extraction
        pycolmap.extract_features(
            database_path=str(self.config.database_path),
            image_path=str(self.config.image_dir),
            sift_options=sift_options
        )
        print(f"Features extracted successfully. Database saved to: {self.config.database_path}")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Feature Extraction")
        parser.add_argument(
            '--project_dir', type=str, required=True,
            help='Path to the project directory (containing frames, outputs)'
        )
        args = parser.parse_args()

        # Initialize and create directories
        config = Config(args.project_dir)
        config.create_directories()

        # Run feature extraction
        extractor = FeatureExtractor(config)
        extractor.extract_features()
        print("Feature extraction pipeline completed.")

if __name__ == "__main__":
    FeatureExtractor.main()
