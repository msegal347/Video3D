import os
import torch
import numpy as np
import open3d as o3d

class PostProcessor:
    def __init__(self, pipeline, bbox_min=(-1.0, -1.0, -1.0), bbox_max=(1.0, 1.0, 1.0),
                 resolution=128, batch_size=1024, density_threshold=0.5):
        """
        PostProcessor for extracting a 3D point cloud from a TensoRF volume.

        Args:
            pipeline (FastSRNeRF): Trained pipeline containing TensoRF and EDSR (EDSR unused here).
            bbox_min (tuple): Min corner of the bounding box (x_min, y_min, z_min).
            bbox_max (tuple): Max corner of the bounding box (x_max, y_max, z_max).
            resolution (int): Resolution along each axis for the 3D grid.
            batch_size (int): Number of points processed in a single batch.
            density_threshold (float): Density threshold for filtering points.
        """
        self.pipeline = pipeline
        self.bbox_min = np.array(bbox_min, dtype=np.float32)
        self.bbox_max = np.array(bbox_max, dtype=np.float32)
        self.resolution = resolution
        self.batch_size = batch_size
        self.density_threshold = density_threshold

        # Determine device
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("PostProcessor using Apple MPS device.")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("PostProcessor using CUDA device.")
        else:
            self.device = torch.device("cpu")
            print("PostProcessor using CPU device.")

        self.pipeline.to(self.device)
        self.pipeline.eval()

    def _generate_3d_grid(self):
        """
        Generate a 3D grid of points within the specified bounding box.
        
        Returns:
            torch.Tensor: [N, 3] tensor of sampled 3D points.
        """
        xs = torch.linspace(self.bbox_min[0], self.bbox_max[0], self.resolution)
        ys = torch.linspace(self.bbox_min[1], self.bbox_max[1], self.resolution)
        zs = torch.linspace(self.bbox_min[2], self.bbox_max[2], self.resolution)
        X, Y, Z = torch.meshgrid(xs, ys, zs, indexing='ij')
        points = torch.stack([X, Y, Z], dim=-1).reshape(-1, 3)  # [N, 3]
        return points.to(self.device)

    def _query_density_color(self, points):
        """
        Given a set of points, query the density and color from the TensoRF model.

        Args:
            points (torch.Tensor): [N, 3] points in world coordinates.

        Returns:
            densities (torch.Tensor): [N] densities at the given points.
            colors (torch.Tensor): [N, 3] colors at the given points.
        """
        N = points.shape[0]
        samples = points.unsqueeze(1)  # [N, 1, 3]

        # Sample features
        features = self.pipeline.nerf.sample_features(samples)  # [N,1,32]
        features = features.squeeze(1)  # [N,32]

        # Query densities
        densities = self.pipeline.nerf.density_mlp(features).squeeze(-1)  # [N]

        # For color, we need a viewing direction. Let's pick a fixed direction:
        # e.g. looking along -Z axis.
        view_dir = torch.tensor([0.0, 0.0, -1.0], device=points.device).unsqueeze(0).expand(N, 3)
        # Combine features and directions
        color_input = torch.cat([features, view_dir], dim=-1)  # [N, 35]
        colors = self.pipeline.nerf.color_mlp(color_input)  # [N, 3]

        return densities, colors

    def extract_point_cloud(self):
        """
        Extract a point cloud by thresholding density on a sampled 3D grid.
        
        Returns:
            o3d.geometry.PointCloud: The extracted 3D point cloud.
        """
        points = self._generate_3d_grid()
        N = points.shape[0]

        all_points = []
        all_colors = []

        with torch.no_grad():
            for start in range(0, N, self.batch_size):
                end = min(start + self.batch_size, N)
                batch_points = points[start:end]
                densities, colors = self._query_density_color(batch_points)

                mask = densities > self.density_threshold
                valid_points = batch_points[mask].cpu().numpy()
                valid_colors = colors[mask].cpu().numpy()

                all_points.append(valid_points)
                all_colors.append(valid_colors)

        if len(all_points) == 0:
            print("No points found above the density threshold.")
            return o3d.geometry.PointCloud()

        all_points = np.concatenate(all_points, axis=0)
        all_colors = np.concatenate(all_colors, axis=0)
        # Normalize colors for visualization
        all_colors = np.clip(all_colors, 0, 1)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(all_points)
        pcd.colors = o3d.utility.Vector3dVector(all_colors)

        return pcd

    def save_point_cloud(self, pcd, save_path):
        """
        Save the generated point cloud to a file.

        Args:
            pcd (o3d.geometry.PointCloud): The point cloud to save.
            save_path (str): Path to save the point cloud file.
        """
        o3d.io.write_point_cloud(save_path, pcd)

    def read_point_cloud(self, file_path):
        """
        Load a point cloud file and visualize it.

        Args:
            file_path (str): Path to the .ply file to load.
        """
        print(f"Loading point cloud from {file_path}...")
        pcd = o3d.io.read_point_cloud(file_path)
        if pcd.is_empty():
            print("Error: The point cloud file is empty or invalid.")
        else:
            print(f"Point cloud loaded. Total points: {len(pcd.points)}")
            o3d.visualization.draw_geometries([pcd])


if __name__ == "__main__":
    from fastsr_nerf_pipeline import FastSRNeRF, TensoRF, EDSR

    # Determine device for loading the pipeline
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS device for PostProcessor main.")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA device for PostProcessor main.")
    else:
        device = torch.device("cpu")
        print("Using CPU for PostProcessor main.")

    # Load trained models
    pipeline = FastSRNeRF()
    pipeline.nerf.load_state_dict(torch.load("checkpoints/trained_nerf_weights.pth", map_location=device))
    pipeline.sr.load_state_dict(torch.load("checkpoints/trained_edsr_weights.pth", map_location=device))
    pipeline.to(device).eval()

    # Initialize PostProcessor with desired bounding box and resolution
    post_processor = PostProcessor(pipeline, bbox_min=(-1, -1, -1), bbox_max=(1, 1, 1), resolution=128, density_threshold=0.5)

    # Extract and save the point cloud
    pcd = post_processor.extract_point_cloud()
    post_processor.save_point_cloud(pcd, save_path="rendered_point_cloud.ply")
    print("Point cloud saved to rendered_point_cloud.ply")

    # Read and visualize the saved point cloud
    post_processor.read_point_cloud("rendered_point_cloud.ply")
