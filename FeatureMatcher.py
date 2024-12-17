import pycolmap
import argparse
from pathlib import Path

class Config:
    def __init__(self, project_dir):
        self.project_dir = Path(project_dir) if project_dir else Path.cwd()
        self.database_path = self.project_dir / 'database.db'

    def validate_database(self):
        if not self.database_path.exists():
            raise FileNotFoundError(f"Database not found at: {self.database_path}. "
                                    "Ensure features have been extracted first.")

class FeatureMatcher:
    def __init__(self, config):
        self.config = config

    def match_features(self):
        """
        Perform exhaustive feature matching using COLMAP.
        """
        print("Matching features...")

        # Validate the database exists
        self.config.validate_database()

        # Set matching options
        matching_options = pycolmap.ExhaustiveMatchingOptions()

        # Perform feature matching
        pycolmap.match_exhaustive(
            database_path=str(self.config.database_path),
            matching_options=matching_options
        )
        print(f"Feature matching completed. Database updated at: {self.config.database_path}")

    @staticmethod
    def main():
        parser = argparse.ArgumentParser(description="Feature Matching")
        parser.add_argument(
            '--project_dir', type=str, required=True,
            help="Path to the output directory containing the database."
        )
        args = parser.parse_args()

        # Initialize configuration and run feature matching
        config = Config(project_dir=args.project_dir)
        matcher = FeatureMatcher(config)
        matcher.match_features()

if __name__ == "__main__":
    FeatureMatcher.main()
