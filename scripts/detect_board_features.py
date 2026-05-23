"""
Detect hold centers and board edges in calibration frames.

For each frame, outputs a JSON with detected feature pixel coordinates
that the calibration UI can show as clickable snap points.

Usage:
    python scripts/detect_board_features.py
    python scripts/detect_board_features.py --force
    python scripts/detect_board_features.py --preview data/calibration_frames/climb_001.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def detect_hold_centers(gray: np.ndarray) -> list[list[float]]:
    """
    Detect hold bolt holes as small dark circles.

    Args:
        gray: Grayscale image.

    Returns:
        List of [x, y] pixel coordinates for detected centers.
    """
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 0
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 800
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.7
    params.filterByInertia = True
    params.minInertiaRatio = 0.4

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(blurred)

    return [[round(kp.pt[0], 1), round(kp.pt[1], 1)] for kp in keypoints]


def detect_features(frame_path: Path) -> dict:
    """Run all detection on a single frame."""
    img = cv2.imread(str(frame_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hold_centers = detect_hold_centers(gray)

    return {
        "frame": frame_path.name,
        "image_size": [img.shape[1], img.shape[0]],
        "board_corners": {"main": None, "kick": None},
        "hold_centers": {"main": hold_centers, "kick": hold_centers},
    }


def preview(frame_path: Path):
    """Show detection results visually for debugging."""
    img = cv2.imread(str(frame_path))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    hold_centers = detect_hold_centers(gray)

    display = img.copy()
    for (x, y) in hold_centers:
        cv2.circle(display, (int(x), int(y)), 5, (0, 255, 0), 1)

    print(f"Hold centers: {len(hold_centers)} detected")

    cv2.namedWindow("Features", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Features", 1280, 720)
    cv2.imshow("Features", display)
    print("Press any key in the image window to close.")
    while True:
        key = cv2.waitKey(100) & 0xFF
        if key != 255:
            break
        if cv2.getWindowProperty("Features", cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--preview", type=Path, default=None,
                        help="Preview detections on a single frame")
    args = parser.parse_args()

    if args.preview:
        preview(args.preview)
        return

    frames = sorted(config.CALIBRATION_FRAMES_DIR.rglob("*.jpg"))
    print(f"Found {len(frames)} calibration frames")

    for frame_path in frames:
        rel = frame_path.relative_to(config.CALIBRATION_FRAMES_DIR)
        out_path = config.DETECTED_FEATURES_DIR / rel.parent / f"{frame_path.stem}_features.json"

        if out_path.exists() and not args.force:
            continue

        features = detect_features(frame_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(features, f, indent=2)
        print(f"  {rel}: {len(features['hold_centers']['main'])} holds detected")

    print("Done.")


if __name__ == "__main__":
    main()