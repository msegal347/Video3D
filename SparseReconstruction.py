import pycolmap
import argparse
from pathlib import Path

class Config:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.image_dir = self.project_dir / 'images'
        self.database_path = self.project_dir / 'database.db'
        self.sparse_dir = self.project_dir / 'sparse'
        self.num_threads = 8  # Optimize for multi-threading

    def validate_inputs(self):
        if not self.image_dir.exists() or not any(self.image_dir.iterdir()):
            raise FileNotFoundError(f"Image directory not found or empty: {self.image_dir}")
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.database_path}")
        print("Input validation passed.")

    def create_directories(self):
        """Create necessary directories for sparse reconstruction."""
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        print(f"Sparse output directory: {self.sparse_dir}")

class SparseReconstruction:
    def __init__(self, config):
        self.config = config

    def incremental_mapping(self):
        """
        Perform incremental Structure-from-Motion (SfM) for sparse reconstruction.
        """
        print("Starting incremental sparse reconstruction...")

        reconstructions = pycolmap.incremental_mapping(
            database_path=str(self.config.database_path),
            image_path=str(self.config.image_dir),
            output_path=str(self.config.sparse_dir),
            options=pycolmap.IncrementalPipelineOptions()
        )

        if reconstructions:
            ply_output = self.config.sparse_dir / "reconstruction.ply"
            reconstructions[0].export_PLY(str(ply_output))
            print(f"Incremental sparse reconstruction completed. Output saved to: {ply_output}")
        else:
            print("No reconstructions were successful. Please verify input data.")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Sparse Reconstruction")
        parser.add_argument(
            '--project_dir', type=str, required=True,
            help="Path to the output directory containing images and database."
        )
        args = parser.parse_args()

        # Initialize and validate the configuration
        config = Config(args.project_dir)
        config.validate_inputs()
        config.create_directories()

        # Run sparse reconstruction
        reconstructor = SparseReconstruction(config)
        reconstructor.incremental_mapping()

if __name__ == "__main__":
    SparseReconstruction.main()
