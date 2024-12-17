from pathlib import Path
import argparse

class Config:
    def __init__(self, project_dir=None):
        # Set the project directory to the provided path or the current working directory
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.image_dir = self.project_dir / 'images'
        self.database_path = self.project_dir / 'database.db'
        self.sparse_dir = self.project_dir / 'sparse'
        self.dense_dir = self.project_dir / 'dense'
        self.num_threads = 4
        self.max_image_size = 1600
        self.first_octave = 0
        self.use_gpu = True  # Set to True if GPU is available and configured

    def create_directories(self):
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.sparse_dir.mkdir(exist_ok=True)
        self.dense_dir.mkdir(exist_ok=True)

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Configuration Setup")
        parser.add_argument('--project_dir', type=str, help='Path to the project directory (defaults to cwd)')
        args = parser.parse_args()

        config = Config(args.project_dir)
        config.create_directories()
        print(f"Directories created in {config.project_dir}")

if __name__ == "__main__":
    Config.main()
