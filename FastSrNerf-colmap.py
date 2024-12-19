import argparse
import os
from pathlib import Path, PurePosixPath
import numpy as np
import json
import cv2
import shutil
from glob import glob
import math
import sys

class ColmapToNerf:
    def __init__(self, args):
        self.args = args
        self.AABB_SCALE = int(args.aabb_scale)

    @staticmethod
    def do_system(command):
        print(f"Executing: {command}")
        result = os.system(command)
        if result != 0:
            raise RuntimeError(f"Command failed: {command}")

    def run_ffmpeg(self):
        ffmpeg_binary = "ffmpeg"
        images_dir = Path(self.args.images).resolve()
        video_file = Path(self.args.video_in).resolve()

        print(f"Converting video {video_file} to images in {images_dir}")
        if not self.args.overwrite and images_dir.exists():
            response = input(f"Folder {images_dir} will be replaced. Continue? (Y/n): ").strip().lower()
            if response != "y":
                raise RuntimeError("Operation cancelled by user.")

        shutil.rmtree(images_dir, ignore_errors=True)
        images_dir.mkdir(parents=True, exist_ok=True)

        time_slice = ""
        if self.args.time_slice:
            start, end = map(float, self.args.time_slice.split(","))
            time_slice = f",select='between(t\\,{start}\\,{end})'"

        self.do_system(
            f"{ffmpeg_binary} -i {video_file} -qscale:v 1 -qmin 1 -vf \"fps={self.args.video_fps}{time_slice}\" {images_dir}/%04d.jpg"
        )

    def run_colmap(self):
        colmap_binary = "colmap"
        images_dir = Path(self.args.images).resolve()
        database_path = Path(self.args.colmap_db).resolve()
        db_noext = str(database_path.with_suffix(""))
        sparse_dir = Path(f"{db_noext}_sparse").resolve()
        text_dir = Path(self.args.text).resolve()

        if self.args.text == "text":
            text_dir = Path(db_noext + "_text")

        print(f"Running COLMAP with images: {images_dir}")

        if not self.args.overwrite and (sparse_dir.exists() or text_dir.exists()):
            response = input(f"Folders {sparse_dir} and {text_dir} will be replaced. Continue? (Y/n): ").strip().lower()
            if response != "y":
                raise RuntimeError("Operation cancelled by user.")

        if database_path.exists():
            database_path.unlink()

        sparse_dir.mkdir(parents=True, exist_ok=True)
        if text_dir.exists():
            shutil.rmtree(text_dir)
        text_dir.mkdir(parents=True, exist_ok=True)

        # Handle camera params if provided
        camera_param_str = f"--ImageReader.camera_params \"{self.args.colmap_camera_params}\"" if self.args.colmap_camera_params else ""

        self.do_system(
            f"{colmap_binary} feature_extractor --ImageReader.camera_model {self.args.colmap_camera_model} {camera_param_str} "
            f"--SiftExtraction.estimate_affine_shape=true --SiftExtraction.domain_size_pooling=true --ImageReader.single_camera 1 "
            f"--database_path {database_path} --image_path {images_dir}"
        )

        match_cmd = f"{colmap_binary} {self.args.colmap_matcher}_matcher --SiftMatching.guided_matching=true --database_path {database_path}"
        if self.args.vocab_path:
            match_cmd += f" --VocabTreeMatching.vocab_tree_path {self.args.vocab_path}"
        self.do_system(match_cmd)

        self.do_system(f"{colmap_binary} mapper --database_path {database_path} --image_path {images_dir} --output_path {sparse_dir}")
        # Optional bundle adjust refine
        self.do_system(f"{colmap_binary} bundle_adjuster --input_path {sparse_dir}/0 --output_path {sparse_dir}/0 --BundleAdjustment.refine_principal_point 1")

        # Convert to text
        self.do_system(f"{colmap_binary} model_converter --input_path {sparse_dir}/0 --output_path {text_dir} --output_type TXT")

    @staticmethod
    def variance_of_laplacian(image):
        return cv2.Laplacian(image, cv2.CV_64F).var()

    def compute_sharpness(self, image_path):
        image = cv2.imread(str(image_path))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return self.variance_of_laplacian(gray)

    @staticmethod
    def qvec_to_rotmat(qvec):
        w, x, y, z = -qvec  # Note the minus sign to match original code logic
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z,       2*x*z + 2*w*y],
            [2*x*y + 2*w*z,       1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x],
            [2*x*z - 2*w*y,       2*y*z + 2*w*x,       1 - 2*x**2 - 2*y**2]
        ])

    @staticmethod
    def closest_point_2_lines(oa, da, ob, db):
        da = da / np.linalg.norm(da)
        db = db / np.linalg.norm(db)
        c = np.cross(da, db)
        denom = np.linalg.norm(c)**2
        t = ob - oa
        ta = np.linalg.det([t, db, c]) / (denom + 1e-10)
        tb = np.linalg.det([t, da, c]) / (denom + 1e-10)
        # force ta and tb <= 0 as in original code
        if ta > 0:
            ta = 0
        if tb > 0:
            tb = 0
        return (oa+ta*da+ob+tb*db) * 0.5, denom

    @staticmethod
    def rotmat(a, b):
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        v = np.cross(a, b)
        c = np.dot(a, b)
        if c < -1 + 1e-10:
            # If they are nearly opposite, perturb 'a'
            return ColmapToNerf.rotmat(a + np.random.uniform(-1e-2, 1e-2, 3), b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]],
                         [v[2], 0, -v[0]],
                         [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat)*((1 - c)/(s**2 + 1e-10))

    def parse_cameras(self):
        cameras_file = Path(self.args.text) / "cameras.txt"
        cameras = {}
        with open(cameras_file, "r") as f:
            for line in f:
                if line.startswith("#"):
                    continue
                els = line.split()
                camera_id = int(els[0])
                model = els[1]
                w = float(els[2])
                h = float(els[3])

                camera = {
                    "w": w,
                    "h": h,
                    "fl_x": None,
                    "fl_y": None,
                    "cx": w/2,
                    "cy": h/2,
                    "k1": 0, "k2": 0, "k3": 0, "k4": 0,
                    "p1": 0, "p2": 0,
                    "is_fisheye": False
                }

                # Parse intrinsics depending on model
                # Refer to original code for parameter order
                if model == "SIMPLE_PINHOLE":
                    # params: f, cx, cy
                    f, cx, cy = map(float, els[4:7])
                    camera["fl_x"] = f
                    camera["fl_y"] = f
                    camera["cx"] = cx
                    camera["cy"] = cy
                elif model == "PINHOLE":
                    # params: fx, fy, cx, cy
                    fx, fy, cx, cy = map(float, els[4:8])
                    camera["fl_x"] = fx
                    camera["fl_y"] = fy
                    camera["cx"] = cx
                    camera["cy"] = cy
                elif model == "SIMPLE_RADIAL":
                    # params: f, cx, cy, k1
                    f, cx, cy, k1 = map(float, els[4:8])
                    camera["fl_x"] = f
                    camera["fl_y"] = f
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                elif model == "RADIAL":
                    # params: f, cx, cy, k1, k2
                    f, cx, cy, k1, k2 = map(float, els[4:9])
                    camera["fl_x"] = f
                    camera["fl_y"] = f
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                    camera["k2"] = k2
                elif model == "OPENCV":
                    # params: fx, fy, cx, cy, k1, k2, p1, p2
                    fx, fy, cx, cy, k1, k2, p1, p2 = map(float, els[4:12])
                    camera["fl_x"] = fx
                    camera["fl_y"] = fy
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                    camera["k2"] = k2
                    camera["p1"] = p1
                    camera["p2"] = p2
                elif model == "SIMPLE_RADIAL_FISHEYE":
                    # params: f, cx, cy, k1
                    f, cx, cy, k1 = map(float, els[4:8])
                    camera["fl_x"] = f
                    camera["fl_y"] = f
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                    camera["is_fisheye"] = True
                elif model == "RADIAL_FISHEYE":
                    # params: f, cx, cy, k1, k2
                    f, cx, cy, k1, k2 = map(float, els[4:9])
                    camera["fl_x"] = f
                    camera["fl_y"] = f
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                    camera["k2"] = k2
                    camera["is_fisheye"] = True
                elif model == "OPENCV_FISHEYE":
                    # params: fx, fy, cx, cy, k1, k2, k3, k4
                    fx, fy, cx, cy, k1, k2, k3, k4 = map(float, els[4:12])
                    camera["fl_x"] = fx
                    camera["fl_y"] = fy
                    camera["cx"] = cx
                    camera["cy"] = cy
                    camera["k1"] = k1
                    camera["k2"] = k2
                    camera["k3"] = k3
                    camera["k4"] = k4
                    camera["is_fisheye"] = True
                else:
                    print("Unknown camera model:", model)

                # Compute camera_angle_x, camera_angle_y
                camera["camera_angle_x"] = math.atan(camera["w"]/(camera["fl_x"]*2))*2
                camera["camera_angle_y"] = math.atan(camera["h"]/(camera["fl_y"]*2))*2

                cameras[camera_id] = camera
        return cameras

    def parse_images(self):
        images_file = Path(self.args.text) / "images.txt"
        images = []
        skip_early = int(self.args.skip_early)
        line_count = 0
        with open(images_file, "r") as f:
            for line in f:
                line = line.strip()
                if line.startswith("#"):
                    continue
                line_count += 1
                # images.txt has lines in pairs: first line pose+filename, second line 2D points
                # We only use every second line (odd lines)
                if line_count % 2 == 1:
                    elems = line.split()
                    image_id = int(elems[0])
                    qvec = np.array(tuple(map(float, elems[1:5])))
                    tvec = np.array(tuple(map(float, elems[5:8])))
                    file_path = "_".join(elems[9:])
                    # Skip early images if requested
                    if line_count < skip_early*2:
                        continue
                    images.append({
                        "id": image_id,
                        "qvec": qvec,
                        "tvec": tvec,
                        "file_path": str(Path(self.args.images)/file_path)
                    })
        return images

    def convert_to_nerf(self):
        print("Converting COLMAP output to NeRF format...")
        cameras = self.parse_cameras()
        images = self.parse_images()

        # For simplicity, assume single camera intrinsics if only one camera is present
        # Otherwise, you might need to handle per-frame intrinsics
        if len(cameras) == 1:
            # single camera
            camera = next(iter(cameras.values()))
            out = {
                "camera_angle_x": camera["camera_angle_x"],
                "camera_angle_y": camera["camera_angle_y"],
                "fl_x": camera["fl_x"],
                "fl_y": camera["fl_y"],
                "k1": camera["k1"],
                "k2": camera["k2"],
                "k3": camera["k3"],
                "k4": camera["k4"],
                "p1": camera["p1"],
                "p2": camera["p2"],
                "is_fisheye": camera["is_fisheye"],
                "cx": camera["cx"],
                "cy": camera["cy"],
                "w": camera["w"],
                "h": camera["h"],
                "aabb_scale": self.AABB_SCALE,
                "frames": []
            }
        else:
            # If multiple cameras, just store frames and aabb_scale
            # (You might enhance this to store multiple cameras)
            out = {
                "aabb_scale": self.AABB_SCALE,
                "frames": []
            }

        bottom = np.array([0,0,0,1]).reshape(1,4)
        up = np.zeros(3)

        # Compute poses
        for img in images:
            qvec = img["qvec"]
            tvec = img["tvec"].reshape(3,1)
            R = self.qvec_to_rotmat(qvec)
            m = np.concatenate([np.concatenate([R, tvec], axis=1), bottom], axis=0)
            c2w = np.linalg.inv(m)

            if not self.args.keep_colmap_coords:
                # Adjust coordinate system to match original code logic
                c2w[0:3,2] *= -1
                c2w[0:3,1] *= -1
                c2w = c2w[[1,0,2,3],:]
                c2w[2,:] *= -1
                up += c2w[0:3,1]

            sharpness = self.compute_sharpness(img["file_path"])
            frame = {
                "file_path": str(PurePosixPath(img["file_path"])),
                "sharpness": sharpness,
                "transform_matrix": c2w.tolist()
            }

            # If multiple camera models, could add intrinsics here per frame
            # if img_camera_id in cameras: frame.update(cameras[img_camera_id])

            out["frames"].append(frame)

        nframes = len(out["frames"])

        if self.args.keep_colmap_coords:
            # Flip matrix as in original code
            flip_mat = np.array([
                [1,0,0,0],
                [0,-1,0,0],
                [0,0,-1,0],
                [0,0,0,1]
            ])
            for f in out["frames"]:
                f["transform_matrix"] = (np.array(f["transform_matrix"]) @ flip_mat).tolist()
        else:
            # Reorient the scene
            up = up / np.linalg.norm(up)
            R = self.rotmat(up, np.array([0,0,1]))
            R = np.pad(R,[(0,1),(0,1)], mode='constant')
            R[-1,-1] = 1
            # Apply rotation
            for f in out["frames"]:
                fmat = np.array(f["transform_matrix"])
                f["transform_matrix"] = (R @ fmat).tolist()

            # Compute center of attention
            print("Computing center of attention...")
            totw = 0.0
            totp = np.array([0.0,0.0,0.0])
            for i in range(nframes):
                mf = np.array(out["frames"][i]["transform_matrix"])[0:3,:]
                for j in range(nframes):
                    mg = np.array(out["frames"][j]["transform_matrix"])[0:3,:]
                    p, w = self.closest_point_2_lines(mf[:,3], mf[:,2], mg[:,3], mg[:,2])
                    if w > 0.00001:
                        totp += p*w
                        totw += w
            if totw > 0:
                totp /= totw

            print("Center of attention:", totp)
            for f in out["frames"]:
                fmat = np.array(f["transform_matrix"])
                fmat[0:3,3] -= totp
                f["transform_matrix"] = fmat.tolist()

            # Scale scene
            avglen = 0.
            for f in out["frames"]:
                fmat = np.array(f["transform_matrix"])
                avglen += np.linalg.norm(fmat[0:3,3])
            avglen /= nframes
            print("Avg camera distance from origin:", avglen)
            for f in out["frames"]:
                fmat = np.array(f["transform_matrix"])
                fmat[0:3,3] *= 4.0/avglen
                f["transform_matrix"] = fmat.tolist()

        # Determine output directory based on video input
        if self.args.video_in:
            output_dir = Path(self.args.video_in).parent
            output_file = output_dir / "transforms.json"
        else:
            output_file = Path(self.args.out)

        # Write out transforms file
        with open(output_file, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Transforms saved to {output_file}")

        # If masking is requested
        if len(self.args.mask_categories) > 0:
            self.apply_masks(out)

    def apply_masks(self, out):
        # This replicates the original code's detectron2 logic if needed
        print("Applying masks using detectron2...")
        try:
            import detectron2
        except ModuleNotFoundError:
            try:
                import torch
            except ModuleNotFoundError:
                print("PyTorch is not installed. Cannot apply masking.")
                return
            input("Detectron2 is not installed. Press enter to attempt install.")
            import subprocess
            package = 'git+https://github.com/facebookresearch/detectron2.git'
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            import detectron2

        import torch
        from detectron2.config import get_cfg
        from detectron2 import model_zoo
        from detectron2.engine import DefaultPredictor

        script_folder = Path(os.path.dirname(__file__)) / "scripts"
        category2id = json.load(open(script_folder / "category2id.json", "r"))
        mask_ids = [category2id[c] for c in self.args.mask_categories]

        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
        predictor = DefaultPredictor(cfg)

        for frame in out["frames"]:
            img_path = Path(frame["file_path"])
            img = cv2.imread(str(img_path))
            outputs = predictor(img)
            output_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
            for i in range(len(outputs["instances"])):
                cls_id = outputs["instances"][i].pred_classes.cpu().numpy()[0]
                if cls_id in mask_ids:
                    pred_mask = outputs["instances"][i].pred_masks.cpu().numpy()[0]
                    output_mask = np.logical_or(output_mask, pred_mask)
            mask_name = img_path.parent / f"dynamic_mask_{img_path.stem}.png"
            cv2.imwrite(str(mask_name), (output_mask*255).astype(np.uint8))
            print(f"Mask saved to {mask_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data from COLMAP for NeRF.")
    parser.add_argument("--video_in", default="", help="Input video file.")
    parser.add_argument("--video_fps", default=2, type=int, help="Frames per second for video extraction.")
    parser.add_argument("--time_slice", default="", help="Time slice in seconds (e.g., '10,20').")
    parser.add_argument("--run_colmap", action="store_true", help="Run COLMAP before conversion.")
    parser.add_argument("--colmap_matcher", default="sequential", choices=["exhaustive","sequential","spatial","transitive","vocab_tree"], help="COLMAP matcher.")
    parser.add_argument("--colmap_db", default="colmap.db", help="COLMAP database file.")
    parser.add_argument("--colmap_camera_model", default="OPENCV", choices=["SIMPLE_PINHOLE", "PINHOLE", "SIMPLE_RADIAL", "RADIAL", "OPENCV", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE", "OPENCV_FISHEYE"])
    parser.add_argument("--colmap_camera_params", default="", help="Intrinsic parameters for camera model.")
    parser.add_argument("--images", default="images", help="Path to extracted images.")
    parser.add_argument("--text", default="colmap_text", help="Path to COLMAP text output.")
    parser.add_argument("--aabb_scale", default=32, choices=["1","2","4","8","16","32","64","128"], help="AABB scale factor.")
    parser.add_argument("--skip_early", default=0, help="Skip this many initial images.")
    parser.add_argument("--keep_colmap_coords", action="store_true", help="Keep COLMAP coordinate system.")
    parser.add_argument("--out", default="transforms.json", help="Path to save NeRF transforms.")
    parser.add_argument("--vocab_path", default="", help="Vocabulary tree path for COLMAP.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    parser.add_argument("--mask_categories", nargs="*", type=str, default=[], help="Object categories to mask out.")
    args = parser.parse_args()

    converter = ColmapToNerf(args)
    if args.video_in:
        converter.run_ffmpeg()
    if args.run_colmap:
        converter.run_colmap()
    converter.convert_to_nerf()
