import subprocess
from pathlib import Path
import argparse


class Config:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir).resolve()
        self.image_dir = self.project_dir / "images"
        self.database_path = self.project_dir / "database.db"
        self.sparse_dir = self.project_dir / "sparse"
        self.dense_dir = self.project_dir / "dense"

    def create_directories(self):
        """Create directories for image, sparse, and dense outputs if they don't exist."""
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(parents=True, exist_ok=True)
        self.dense_dir.mkdir(parents=True, exist_ok=True)


class DenseReconstruction:
    def __init__(self, config, colmap_path):
        self.config = config
        self.colmap_path = colmap_path  # Path to COLMAP.bat or executable

    def run_colmap_command(self, command):
        """Run a COLMAP command using subprocess."""
        full_command = [str(self.colmap_path)] + command
        print(f"Running COLMAP command: {' '.join(full_command)}")
        try:
            subprocess.run(full_command, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"COLMAP command failed: {' '.join(full_command)}\nError: {e}")

    def undistort_images(self):
        """Undistort images for dense reconstruction."""
        print("Undistorting images...")
        sparse_subdirs = list(self.config.sparse_dir.glob("*"))
        if not sparse_subdirs:
            raise ValueError(f"No subdirectories found in sparse directory: {self.config.sparse_dir}")

        sparse_input_path = sparse_subdirs[0]
        undistort_output = self.config.dense_dir / "undistorted_images"
        undistort_output.mkdir(parents=True, exist_ok=True)

        command = [
            "image_undistorter",
            "--image_path", str(self.config.image_dir),
            "--input_path", str(sparse_input_path),
            "--output_path", str(undistort_output),
            "--output_type", "COLMAP",
        ]
        self.run_colmap_command(command)
        print(f"Images undistorted to: {undistort_output}")

    def patch_match_stereo(self):
        """Run PatchMatch stereo."""
        print("Running PatchMatch stereo...")
        undistort_output = self.config.dense_dir / "undistorted_images"

        command = [
            "patch_match_stereo",
            "--workspace_path", str(undistort_output),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
        ]
        self.run_colmap_command(command)
        print("PatchMatch stereo completed.")

    def stereo_fusion(self):
        """Fuse depth maps into a dense point cloud."""
        print("Fusing depth maps into a dense point cloud...")
        undistort_output = self.config.dense_dir / "undistorted_images"
        point_cloud_output = self.config.dense_dir / "point_cloud.ply"

        command = [
            "stereo_fusion",
            "--workspace_path", str(undistort_output),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(point_cloud_output),
        ]
        self.run_colmap_command(command)
        print(f"Dense point cloud saved to: {point_cloud_output}")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Dense Reconstruction")
        parser.add_argument(
            "--project_dir", type=str, required=True,
            help="Path to the project directory containing images and reconstructions."
        )
        parser.add_argument(
            "--colmap_path", type=Path, required=True,
            help="Path to the COLMAP executable or batch file."
        )
        args = parser.parse_args()

        # Setup configuration and directories
        config = Config(args.project_dir)
        config.create_directories()

        # Start the dense reconstruction pipeline
        reconstructor = DenseReconstruction(config, colmap_path=args.colmap_path)
        try:
            reconstructor.undistort_images()
            reconstructor.patch_match_stereo()
            reconstructor.stereo_fusion()
        except Exception as e:
            print(f"An error occurred during dense reconstruction: {e}")


if __name__ == "__main__":
    DenseReconstruction.main()
