ffmpeg -i input.webm -c:v libx264 -c:a aac output.mp4

python ExtractFrames.py pantheon.mp4

python FeatureExtractor.py --project_dir pantheon_output

python FeatureMatcher.py --project_dir pantheon_output

python SparseReconstruction.py --output_dir pantheon_output

python ReconstructionVisualizer.py --project_dir pantheon_output --type sparse

python DenseReconstruction.py --project_dir pantheon_output --colmap_path colmap_prebuilt\COLMAP.bat

python ReconstructionVisualizer.py --project_dir pantheon_output --type dense