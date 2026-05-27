"""
Pre-warp raw videos into board space for the route editing UI.

For each video with a calibration, applies both homographies (main + kick)
to every frame and writes a stitched board-space video to data/warped/.

Usage:
    python scripts/warp_videos.py
    python scripts/warp_videos.py --force           # re-warp all
    python scripts/warp_videos.py --scale 6         # higher resolution
    python scripts/warp_videos.py --fps 15          # higher frame rate
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config
from pipeline.calibration import load_calibration, compute_homographies, kick_threshold_px

# Board geometry
BOARD_LEFT = 0
BOARD_RIGHT = 144
BOARD_BOTTOM = 0
BOARD_TOP = 156
KICKBOARD_Y = 12


def build_warp_matrices(
    H_main: np.ndarray,
    H_kick: np.ndarray | None,
    scale: int,
    pad: int,
) -> tuple[np.ndarray, np.ndarray | None, tuple[int, int]]:
    """
    Build adjusted homography matrices that warp pixel space directly
    into the output image coordinate system.

    The output image uses the same coordinate convention as visualize.py:
    x increases rightward, y is flipped (board-top at pixel-top).

    Args:
        H_main: 3x3 homography from pixel space to board space (main board).
        H_kick: 3x3 homography from pixel space to board space (kickboard),
            or None if no kickboard calibration.
        scale: Pixels per board unit in the output.
        pad: Padding in board units around the board.

    Returns:
        (H_main_adj, H_kick_adj, (out_w, out_h)) where adjusted matrices
        map directly from source pixels to output pixels.
    """
    out_w = int((BOARD_RIGHT - BOARD_LEFT + 2 * pad) * scale)
    out_h = int((BOARD_TOP - BOARD_BOTTOM + 2 * pad) * scale)

    # S maps board coords -> output pixel coords (with y-flip)
    S = np.array([
        [scale, 0, (pad - BOARD_LEFT) * scale],
        [0, -scale, (BOARD_TOP + pad) * scale],
        [0, 0, 1],
    ], dtype=np.float64)

    H_main_adj = S @ H_main
    H_kick_adj = S @ H_kick if H_kick is not None else None

    return H_main_adj, H_kick_adj, (out_w, out_h)


def warp_video(
    video_path: Path,
    output_path: Path,
    video_stem: str,
    scale: int = config.WARP_SCALE,
    pad: int = 4,
    target_fps: int = config.WARP_FPS,
) -> bool:
    """
    Warp a single video into board space and write to output_path.

    Args:
        video_path: Path to the source video.
        output_path: Path for the output .mp4.
        video_stem: Video filename stem (for loading calibration).
        scale: Pixels per board unit.
        pad: Padding in board units around the board.
        target_fps: Output frame rate (subsamples source frames).

    Returns:
        True if successful, False on error.
    """
    try:
        cal = load_calibration(video_stem)
        H_main, H_kick = compute_homographies(cal)
        threshold = kick_threshold_px(cal)
    except (FileNotFoundError, ValueError) as e:
        print(f"  Skip {video_stem}: {e}")
        return False

    H_main_adj, H_kick_adj, (out_w, out_h) = build_warp_matrices(
        H_main, H_kick, scale, pad
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  Cannot open: {video_path}")
        return False

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Frame sampling: keep every Nth frame to hit target_fps
    frame_interval = max(1, round(src_fps / target_fps))
    effective_fps = src_fps / frame_interval

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        effective_fps,
        (out_w, out_h),
    )

    # Kickboard boundary row in output image
    kick_row = int((BOARD_TOP + pad - KICKBOARD_Y) * scale)

    frame_idx = 0
    written = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % frame_interval == 0:
            warped = cv2.warpPerspective(frame, H_main_adj, (out_w, out_h))

            if H_kick_adj is not None:
                warped_kick = cv2.warpPerspective(frame, H_kick_adj, (out_w, out_h))
                # Composite kick region: overwrite where kick warp has content
                kick_region = warped_kick[kick_row:, :]
                mask = kick_region.any(axis=2)
                warped[kick_row:, :][mask] = kick_region[mask]

            writer.write(warped)
            written += 1

        frame_idx += 1

    cap.release()
    writer.release()
    print(f"  {video_stem}: {written} frames @ {effective_fps:.1f}fps -> {output_path}")
    
    # Re-encode to H.264 for browser playback
    h264_path = output_path.with_suffix(".tmp.mp4")
    os.system(
        f'ffmpeg -y -i "{output_path}" -c:v libx264 -preset fast -crf 23 '
        f'-g 3 -an "{h264_path}" -loglevel error'
    )
    if h264_path.exists():
        h264_path.replace(output_path)
        
    return True


def main():
    parser = argparse.ArgumentParser(description="Pre-warp videos into board space")
    parser.add_argument("--force", action="store_true", help="Re-warp all videos")
    parser.add_argument("--scale", type=int, default=config.WARP_SCALE,
                        help=f"Pixels per board unit (default: {config.WARP_SCALE})")
    parser.add_argument("--fps", type=int, default=config.WARP_FPS,
                        help=f"Output frame rate (default: {config.WARP_FPS})")
    args = parser.parse_args()

    # Find all raw videos that have calibrations
    videos = sorted(
        p for p in config.RAW_VIDEO_DIR.rglob("*")
        if p.suffix.lower() in config.VIDEO_EXTENSIONS
    )

    print(f"Found {len(videos)} raw videos")
    warped = 0
    skipped = 0

    for video_path in videos:
        stem = video_path.stem
        rel = video_path.relative_to(config.RAW_VIDEO_DIR)
        output_path = config.WARPED_DIR / rel.parent / f"{stem}.mp4"

        if output_path.exists() and not args.force:
            skipped += 1
            continue

        if warp_video(video_path, output_path, stem, args.scale, 4, args.fps):
            warped += 1

    print(f"\nWarped {warped} videos, skipped {skipped} existing")


if __name__ == "__main__":
    main()