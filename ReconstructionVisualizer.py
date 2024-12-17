import open3d as o3d
import subprocess
import argparse
from pathlib import Path
import numpy as np


class ReconstructionVisualizer:
    def __init__(self, project_dir, colmap_path="colmap"):
        # Use the provided project directory
        self.project_dir = Path(project_dir).resolve()
        self.sparse_dir = self.project_dir / "sparse"
        self.dense_dir = self.project_dir / "dense"
        self.image_dir = self.project_dir / "images"
        self.colmap_path = colmap_path  # Path to the COLMAP executable/batch file

    def run_colmap_command(self, command):
        """Run a COLMAP command using subprocess."""
        full_command = [str(self.colmap_path)] + command
        print(f"Running COLMAP command: {' '.join(full_command)}")
        try:
            subprocess.run(full_command, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"COLMAP command failed: {' '.join(full_command)}\nError: {e}")

    def generate_dense(self):
        """Generate dense reconstruction using COLMAP."""
        print("Starting dense reconstruction...")

        # Step 1: Undistort images
        print("Undistorting images...")
        sparse_subdirs = list(self.sparse_dir.glob("*"))
        if not sparse_subdirs:
            raise ValueError(f"No subdirectories found in sparse directory: {self.sparse_dir}")
        sparse_input_path = sparse_subdirs[0]  # Use the first sparse reconstruction
        undistort_output = self.dense_dir / "undistorted_images"
        undistort_output.mkdir(parents=True, exist_ok=True)

        self.run_colmap_command([
            "image_undistorter",
            "--image_path", str(self.image_dir),
            "--input_path", str(sparse_input_path),
            "--output_path", str(undistort_output),
            "--output_type", "COLMAP",
        ])
        print(f"Images undistorted to: {undistort_output}")

        # Step 2: PatchMatch stereo
        print("Running PatchMatch stereo...")
        self.run_colmap_command([
            "patch_match_stereo",
            "--workspace_path", str(undistort_output),
            "--workspace_format", "COLMAP",
            "--PatchMatchStereo.geom_consistency", "true",
        ])
        print("PatchMatch stereo completed.")

        # Step 3: Stereo fusion
        print("Fusing depth maps into a dense point cloud...")
        point_cloud_output = self.dense_dir / "point_cloud.ply"
        self.run_colmap_command([
            "stereo_fusion",
            "--workspace_path", str(undistort_output),
            "--workspace_format", "COLMAP",
            "--input_type", "geometric",
            "--output_path", str(point_cloud_output),
        ])
        print(f"Dense point cloud saved to: {point_cloud_output}")

    def flip_point_cloud(self, point_cloud_path, output_path, axis="y"):
        """Flip a point cloud along the specified axis."""
        print(f"Flipping point cloud along {axis}-axis...")
        pcd = o3d.io.read_point_cloud(str(point_cloud_path))

        # Define the transformation matrix
        if axis == 'x':
            flip_matrix = np.diag([-1, 1, 1, 1])
        elif axis == 'y':
            flip_matrix = np.diag([1, -1, 1, 1])
        elif axis == 'z':
            flip_matrix = np.diag([1, 1, -1, 1])
        else:
            raise ValueError("Axis must be 'x', 'y', or 'z'.")

        # Apply the transformation
        pcd.transform(flip_matrix)
        o3d.io.write_point_cloud(str(output_path), pcd)
        print(f"Flipped point cloud saved to: {output_path}")

    def visualize_sparse(self):
        """Visualize the sparse reconstruction."""
        sparse_file = self.sparse_dir / "reconstruction.ply"
        if sparse_file.exists():
            print(f"Loading sparse reconstruction from: {sparse_file}")
            pcd = o3d.io.read_point_cloud(str(sparse_file))
            o3d.visualization.draw_geometries([pcd], window_name="Sparse Reconstruction")
        else:
            print(f"Sparse reconstruction file not found: {sparse_file}")

    def visualize_dense(self, flip_axis="y"):
        """Visualize the dense reconstruction with optional flipping."""
        dense_file = self.dense_dir / "point_cloud.ply"
        flipped_file = self.dense_dir / "flipped_point_cloud.ply"

        if dense_file.exists():
            # Flip and visualize the point cloud
            self.flip_point_cloud(dense_file, flipped_file, axis=flip_axis)
            print(f"Loading flipped dense reconstruction from: {flipped_file}")
            pcd = o3d.io.read_point_cloud(str(flipped_file))
            o3d.visualization.draw_geometries([pcd], window_name="Flipped Dense Reconstruction")
        else:
            print(f"Dense reconstruction file not found: {dense_file}")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Reconstruction Visualization and Dense Generation")
        parser.add_argument(
            "--project_dir", type=str, required=True,
            help="Path to the project directory containing 'images', 'sparse', and 'dense' folders."
        )
        parser.add_argument(
            "--colmap_path", type=str, default="colmap",
            help="Path to the COLMAP executable or batch file (default: 'colmap')"
        )
        parser.add_argument(
            "--type", type=str, choices=["sparse", "dense"], default="sparse",
            help="Type of reconstruction to visualize (default: sparse)"
        )
        parser.add_argument(
            "--generate", action="store_true",
            help="Generate the dense reconstruction if specified."
        )
        parser.add_argument(
            "--flip_axis", type=str, choices=["x", "y", "z"], default="y",
            help="Axis to flip the dense point cloud during visualization (default: 'y')"
        )
        args = parser.parse_args()

        visualizer = ReconstructionVisualizer(args.project_dir, args.colmap_path)

        if args.generate and args.type == "dense":
            visualizer.generate_dense()

        if args.type == "sparse":
            visualizer.visualize_sparse()
        elif args.type == "dense":
            visualizer.visualize_dense(flip_axis=args.flip_axis)


if __name__ == "__main__":
    ReconstructionVisualizer.main()
