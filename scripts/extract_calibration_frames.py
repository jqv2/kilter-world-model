"""
Extract a single reference frame from each video in data/raw/.

Usage:
    python scripts/extract_calibration_frames.py
    python scripts/extract_calibration_frames.py --force
"""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def find_videos(raw_dir: Path) -> list[Path]:
    """Recursively find all videos in raw_dir."""
    return sorted(
        p for p in raw_dir.rglob("*")
        if p.suffix.lower() in config.VIDEO_EXTENSIONS
    )


def extract_frame(video_path: Path, output_path: Path, frame_frac: float = 0.5) -> bool:
    """Extract a single frame at the given fraction of the video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frame_frac))
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"  Failed to read frame: {video_path}")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-extract all frames")
    args = parser.parse_args()

    videos = find_videos(config.RAW_VIDEO_DIR)
    print(f"Found {len(videos)} videos")

    extracted = 0
    for video_path in videos:
        rel = video_path.relative_to(config.RAW_VIDEO_DIR)
        frame_path = config.CALIBRATION_FRAMES_DIR / rel.parent / f"{video_path.stem}.jpg"

        if frame_path.exists() and not args.force:
            continue

        if extract_frame(video_path, frame_path):
            print(f"  Extracted: {rel}")
            extracted += 1

    print(f"\nExtracted {extracted} frames. Total: {len(list(config.CALIBRATION_FRAMES_DIR.rglob('*.jpg')))}")


if __name__ == "__main__":
    main()