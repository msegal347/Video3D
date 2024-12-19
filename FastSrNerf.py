import os
import json
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import numpy as np
import open3d as o3d
from torch.nn.functional import grid_sample
from lpips import LPIPS
from torch.cuda.amp import GradScaler
from tqdm import tqdm

# For loading the VGG16 model with weights instead of pretrained
import torchvision.models as models
from torchvision.models import VGG16_Weights

# Parse command line arguments
parser = argparse.ArgumentParser(description="FastSR-NeRF Training")
parser.add_argument("--image_dir", type=str, required=True, help="Directory containing images and transforms.json")
parser.add_argument("--patch_size", type=int, required=True, help="Patch size for HR patches")
parser.add_argument("--batch_size", type=int, required=True, help="Batch size for training")
parser.add_argument("--epochs", type=int, required=True, help="Number of training epochs")
parser.add_argument("--learning_rate", type=float, required=True, help="Learning rate")
parser.add_argument("--checkpoint_dir", type=str, required=True, help="Directory to save checkpoints")
args = parser.parse_args()

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

class TensoRF(nn.Module):
    def __init__(self, feature_grid_size=64, num_samples=128):
        super(TensoRF, self).__init__()
        self.num_samples = num_samples
        self.register_buffer('t_vals', torch.linspace(0.0, 1.0, self.num_samples))

        self.feature_grid = nn.Parameter(torch.randn((1, 32, feature_grid_size, feature_grid_size, feature_grid_size)) * 0.1)

        self.density_mlp = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        self.color_mlp = nn.Sequential(
            nn.Linear(35, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 3)
        )

    def forward(self, rays):
        origins, directions = rays[:, :3], rays[:, 3:6]
        t_vals = self.t_vals.unsqueeze(0).expand(rays.size(0), self.num_samples)

        samples = origins.unsqueeze(1) + t_vals.unsqueeze(-1)*directions.unsqueeze(1)
        features = self.sample_features(samples)
        densities = self.density_mlp(features).squeeze(-1)
        directions_expanded = directions.unsqueeze(1).expand(features.shape[0], features.shape[1], 3)
        colors = self.color_mlp(torch.cat([features, directions_expanded], dim=-1))

        weights = self.compute_weights(densities, t_vals)
        rendered_colors = torch.sum(weights.unsqueeze(-1)*colors, dim=1)
        return rendered_colors

    def sample_features(self, samples):
        size = self.feature_grid.shape[-1]
        grid_coords = 2.0*(samples/(size-1)) - 1.0

        N, S, _ = samples.shape
        flat_samples = samples.view(1, 1, N*S, 1, 3)
        flat_grid = grid_sample(self.feature_grid, flat_samples, align_corners=True)

        features = flat_grid.view(32, N, S).permute(1, 2, 0)
        return features

    def compute_weights(self, densities, t_vals):
        deltas = t_vals[:, 1:] - t_vals[:, :-1]
        infinite_delta = torch.tensor([1e10], device=deltas.device).expand(deltas.shape[0], 1)
        deltas = torch.cat([deltas, infinite_delta], dim=-1)

        alphas = 1.0 - torch.exp(-densities*deltas)
        transmittance = torch.cumprod(1.0 - alphas + 1e-10, dim=-1)
        trans_shifted = torch.cat([torch.ones_like(transmittance[:, :1]), transmittance[:, :-1]], dim=-1)
        weights = alphas * trans_shifted
        return weights

class EDSR(nn.Module):
    def __init__(self, upscale_factor=2):
        super(EDSR, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.res_blocks = nn.Sequential(
            *[self._res_block(64) for _ in range(8)]
        )
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Sequential(
            nn.Conv2d(64, 64*(upscale_factor**2), kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(upscale_factor),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def _res_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, lr_images):
        x = self.relu(self.conv1(lr_images))
        residual = x
        x = self.res_blocks(x) + residual
        x = self.conv2(x)
        hr_images = self.upsample(x)
        return hr_images

class NeRFDataset(Dataset):
    def __init__(self, image_dir, patch_size):
        self.image_dir = image_dir
        self.patch_size = patch_size

        transforms_path = os.path.join(image_dir, "transforms.json")
        with open(transforms_path, "r") as f:
            transforms_data = json.load(f)

        self.images = []
        self.transform_matrices = []

        for frame in transforms_data["frames"]:
            rel_path = frame["file_path"].replace("\\", "/")
            if rel_path.startswith("images/"):
                rel_path = rel_path[len("images/"):]
            file_path = os.path.normpath(os.path.join(image_dir, rel_path))

            if os.path.exists(file_path):
                self.images.append(file_path)
                self.transform_matrices.append(np.array(frame["transform_matrix"]))
            else:
                print(f"Warning: File {file_path} does not exist!")

        if not self.images:
            raise ValueError("No valid images found in the specified directory and transforms.json.")

        print(f"Found {len(self.images)} valid images in {image_dir}")

        self.intrinsics = {
            "fl_x": transforms_data["fl_x"],
            "fl_y": transforms_data["fl_y"],
            "cx": transforms_data["cx"],
            "cy": transforms_data["cy"],
            "w": transforms_data["w"],
            "h": transforms_data["h"]
        }

        self.image_tensors = [ToTensor()(np.asarray(o3d.io.read_image(img_path))) for img_path in self.images]

        if len(self.image_tensors) == 0:
            raise ValueError("No valid images found in the specified directory and transforms.json.")

    def __len__(self):
        return len(self.image_tensors)

    def __getitem__(self, idx):
        image = self.image_tensors[idx]
        c, h, w = image.shape

        y = np.random.randint(0, h - self.patch_size)
        x = np.random.randint(0, w - self.patch_size)

        patch_hr = image[:, y:y+self.patch_size, x:x+self.patch_size]
        patch_lr = nn.functional.interpolate(
            patch_hr.unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False
        ).squeeze(0)

        yy, xx = torch.meshgrid(
            torch.arange(y, y+self.patch_size),
            torch.arange(x, x+self.patch_size),
            indexing='ij'
        )
        yy = yy.flatten().float()
        xx = xx.flatten().float()

        fl_x = self.intrinsics["fl_x"]
        fl_y = self.intrinsics["fl_y"]
        cx = self.intrinsics["cx"]
        cy = self.intrinsics["cy"]

        X_cam = (xx - cx) / fl_x
        Y_cam = (yy - cy) / fl_y

        directions_cam = torch.stack([X_cam, Y_cam, -torch.ones_like(X_cam)], dim=-1)

        transform = self.transform_matrices[idx]
        R = transform[:3, :3]
        T = transform[:3, 3]

        directions_world = directions_cam @ torch.tensor(R, dtype=torch.float32).T
        directions_world = directions_world / torch.norm(directions_world, dim=-1, keepdim=True)

        origins_world = torch.tensor(T, dtype=torch.float32).unsqueeze(0).expand_as(directions_world)
        rays = torch.cat([origins_world, directions_world], dim=-1)

        return rays, patch_lr, patch_hr

class FastSRNeRF(nn.Module):
    def __init__(self, nerf_model, sr_model):
        super(FastSRNeRF, self).__init__()
        self.nerf = nerf_model
        self.sr = sr_model

    def forward(self, rays, lr_images):
        hr_images = self.sr(lr_images)
        return hr_images

if __name__ == "__main__":
    image_dir = args.image_dir
    patch_size = args.patch_size
    batch_size = args.batch_size
    epochs = args.epochs
    learning_rate = args.learning_rate
    checkpoint_dir = args.checkpoint_dir

    os.makedirs(checkpoint_dir, exist_ok=True)

    nerf = TensoRF().to(device)
    sr = EDSR(upscale_factor=2).to(device)
    pipeline = FastSRNeRF(nerf, sr).to(device)

    dataset = NeRFDataset(image_dir, patch_size)

    if len(dataset) == 0:
        raise ValueError("Dataset is empty. Check your transforms.json and image directory.")

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    optimizer = optim.Adam(pipeline.parameters(), lr=learning_rate)
    criterion_mse = nn.MSELoss()

    # Create LPIPS without pretrained to avoid warning
    criterion_perceptual = LPIPS(net='vgg', pretrained=False).to(device)
    # Replace LPIPS's net with a weights-based model
    vgg_model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
    criterion_perceptual.net = vgg_model
    criterion_perceptual.eval()

    scaler = GradScaler()

    print("Starting warm-up phase...")
    warmup_iters = 1000
    pipeline.train()
    for _ in range(warmup_iters):
        for rays, lr_patches, hr_patches in tqdm(dataloader, desc="Warm-up"):
            lr_patches = lr_patches.to(device, non_blocking=True)
            rays = rays.to(device, non_blocking=True)
            with torch.amp.autocast(device_type='cuda'):
                pipeline.nerf(rays.view(-1,6))
    print("Warm-up complete. Starting end-to-end training...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        pipeline.train()
        for rays, lr_patches, hr_patches in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            rays = rays.to(device, non_blocking=True)
            lr_patches = lr_patches.to(device, non_blocking=True)
            hr_patches = hr_patches.to(device, non_blocking=True)

            rays_flat = rays.view(-1, 6)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type='cuda'):
                sr_outputs = pipeline(rays_flat, lr_patches)
                loss_mse = criterion_mse(sr_outputs, hr_patches)
                loss_perceptual = criterion_perceptual(sr_outputs, hr_patches).mean()
                total_loss = loss_mse + loss_perceptual

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += total_loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / len(dataloader):.4f}")

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pth")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': pipeline.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': epoch_loss
        }, checkpoint_path)

    print("Training Complete.")

    torch.save(pipeline.nerf.state_dict(), os.path.join(checkpoint_dir, "trained_nerf_weights.pth"))
    torch.save(pipeline.sr.state_dict(), os.path.join(checkpoint_dir, "trained_edsr_weights.pth"))



# python FastSrNerf.py --image_dir "D:/Work/Video3D/images" --patch_size 64 --batch_size 8 --epochs 20 --learning_rate 0.0001 --checkpoint_dir "checkpoints"
