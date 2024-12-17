import ffmpeg
from pathlib import Path

class FrameExtractor:
    def __init__(self, video_path, fps=1):
        """
        Initialize the FrameExtractor with video path and FPS.

        Args:
            video_path (str): Path to the input video file.
            fps (int): Frames per second to extract. Defaults to 1.
        """
        self.video_path = Path(video_path)
        self.fps = fps
        self.output_dir = Path(__file__).resolve().parent.parent / f"{self.video_path.stem}_output" / "images"

    def extract_frames(self):
        """
        Extract frames from the video using FFmpeg.

        Returns:
            Path: The directory where frames are saved.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Define the output pattern for frames
        output_pattern = self.output_dir / "frame_%04d.png"

        # Run FFmpeg to extract frames
        (
            ffmpeg
            .input(str(self.video_path))
            .output(str(output_pattern), vf=f'fps={self.fps}')
            .run(overwrite_output=True)
        )
        print(f"Frames extracted to: {self.output_dir}")
        return self.output_dir

    @staticmethod
    def main():
        import argparse

        parser = argparse.ArgumentParser(description="Extract frames from a video")
        parser.add_argument("video_path", type=str, help="Path to the input video file")
        parser.add_argument("--fps", type=int, default=1, help="Frames per second to extract (default: 1)")
        args = parser.parse_args()

        # Initialize and run the frame extraction
        extractor = FrameExtractor(args.video_path, args.fps)
        extractor.extract_frames()


if __name__ == "__main__":
    FrameExtractor.main()
