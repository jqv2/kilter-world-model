"""
Dataset construction: merge pose sequences with calibrations and route
definitions into training-ready data for the world model.

Pipeline per video:
    1. Load pose JSON
    2. Clean missing frames (interpolation / forward-fill)
    3. Load calibration, transform keypoints to board space
    4. Look up route holds from Kilter DB via climb_log.csv
    5. Package as training sequences with route context

The output includes both pose sequences and per-video route hold arrays.
"""

import csv
import json
import sqlite3
from pathlib import Path

import numpy as np

import config
from pipeline.pose_cleaning import clean_pose_sequence, clean_board_space_poses
from pipeline.calibration import (
    load_calibration,
    transform_pose_sequence,
)
from pipeline.routes import (
    apply_route_edits,
    decode_frames_string,
    holds_to_array,
    lookup_route_by_uuid,
    lookup_route_by_name,
)


def load_dataset(path: Path) -> dict:
    """
    Load a dataset saved by build_dataset.py.

    Args:
        path: Path to the .npz file.

    Returns:
        Dict with train/test sequences, scores, holds, roles, stems, and fps.
    """
    raw = np.load(path, allow_pickle=True)

    def unpack(arr):
        return [np.array(x, dtype=np.float32) for x in arr]

    def unpack_int(arr):
        return [np.array(x, dtype=np.int64) for x in arr]

    return {
        "train_sequences": unpack(raw["train_sequences"]),
        "train_scores": unpack(raw["train_scores"]),
        "train_holds": unpack(raw["train_holds"]),
        "train_roles": unpack_int(raw["train_roles"]),
        "train_route_holds": list(raw["train_route_holds"]),
        "test_sequences": unpack(raw["test_sequences"]),
        "test_scores": unpack(raw["test_scores"]),
        "test_holds": unpack(raw["test_holds"]),
        "test_roles": unpack_int(raw["test_roles"]),
        "test_route_holds": list(raw["test_route_holds"]),
        "train_stems": list(raw["train_stems"]),
        "test_stems": list(raw["test_stems"]),
        "fps": float(raw["fps"]),
    }


def find_paired_videos(
    poses_dir: Path | None = None,
    calibrations_dir: Path | None = None,
) -> list[tuple[Path, str]]:
    """
    Find videos that have both a pose JSON and a calibration file.

    Args:
        poses_dir: Directory containing pose JSONs. Defaults to config.POSES_DIR.
        calibrations_dir: Directory containing calibrations.
            Defaults to config.CALIBRATIONS_DIR.

    Returns:
        List of (pose_json_path, video_stem) tuples for videos with both files.
    """
    p_dir = poses_dir or config.POSES_DIR
    c_dir = calibrations_dir or config.CALIBRATIONS_DIR

    pose_jsons = sorted(
        p for p in p_dir.rglob("*.json")
        if not p.stem.endswith("_overlay")
    )

    paired = []
    for pj in pose_jsons:
        stem = pj.stem
        cal_matches = list(c_dir.rglob(f"{stem}_calibration.json"))
        if cal_matches:
            paired.append((pj, stem))

    return paired


def load_climb_log(log_path: Path | None = None) -> dict[str, dict]:
    """
    Load climb_log.csv into a dict keyed by filename stem.

    Returns:
        Dict mapping video stem -> row dict with keys like
        'route_name', 'grade', 'climb_uuid', etc.
    """
    log_path = log_path or (config.RAW_VIDEO_DIR / "climb_log.csv")
    if not log_path.exists():
        return {}

    log = {}
    with open(log_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename", "")
            stem = Path(filename).stem
            if stem:
                log[stem] = dict(row)
    return log


def resolve_route(
    climb_log_entry: dict | None,
    db_path: Path | None = None,
) -> list[dict] | None:
    """
    Resolve a climb_log entry to a list of route holds.

    Tries climb_uuid first, then falls back to route_name matching.

    Args:
        climb_log_entry: Row dict from climb_log.csv (may have 'climb_uuid'
            and/or 'route_name').
        db_path: Path to kilter.db.

    Returns:
        List of hold dicts, or None if the route can't be resolved.
    """
    if climb_log_entry is None:
        return None

    db = db_path or (config.DATA_DIR / "kilter.db")

    # Try UUID first
    uuid = climb_log_entry.get("climb_uuid", "").strip()
    if uuid:
        result = lookup_route_by_uuid(uuid, db)
        if result:
            return result["holds"]

    # Fall back to name
    name = climb_log_entry.get("route_name", "").strip()
    if name:
        result = lookup_route_by_name(name, db)
        if result:
            return result["holds"]

    return None


def process_video(
    pose_json_path: Path,
    video_stem: str,
    climb_log: dict[str, dict],
    normalize_holds: bool = True,
    apply_edits: bool = True,
) -> dict | None:
    """
    Run the full cleaning + calibration + route lookup + hold filtering pipeline for a single video.

    Args:
        pose_json_path: Path to the pose extraction JSON.
        video_stem: Video filename without extension.
        climb_log: Dict from load_climb_log().
        normalize_holds: If True, normalize hold positions to [0, 1].
        apply_edits: If True, apply route edit exclusions from
            data/route_edits/. If False, use raw DB holds.

    Returns:
        Dict with:
            'keypoints': np.ndarray (T, 17, 2) in board space
            'scores': np.ndarray (T, 17)
            'hold_positions': np.ndarray (N, 2) route hold positions
            'hold_roles': np.ndarray (N,) role IDs per hold
            'fps': float
            'video_stem': str
            'route_name': str or None
        Or None if the video has insufficient valid frames or no route.
    """
    with open(pose_json_path) as f:
        data = json.load(f)

    fps = data["fps"]
    frames = data["frames"]

    # Step 1: Clean missing frames in pixel space
    frames = clean_pose_sequence(frames)

    # Step 2: Transform to board space
    calibration = load_calibration(video_stem)
    frames = transform_pose_sequence(frames, calibration)

    # Step 3: Extract valid frames
    keypoints = []
    scores = []
    for frame in frames:
        if frame["keypoints"] is not None:
            keypoints.append(frame["keypoints"])
            scores.append(frame["scores"])

    if len(keypoints) < 2:
        return None

    # Step 4: Clean keypoint jumps in board space
    keypoints = np.array(keypoints, dtype=np.float32)
    keypoints = np.array(clean_board_space_poses(list(keypoints)), dtype=np.float32)

    # Step 5: Look up route
    log_entry = climb_log.get(video_stem)
    holds = resolve_route(log_entry)
    if holds is None:
        return None

    if apply_edits:
        holds = apply_route_edits(holds, video_stem)

    hold_positions, hold_roles = holds_to_array(holds, normalize=normalize_holds)

    return {
        "keypoints": keypoints,
        "scores": np.array(scores, dtype=np.float32),
        "hold_positions": hold_positions,
        "hold_roles": hold_roles,
        "route_holds_raw": holds,
        "fps": fps,
        "video_stem": video_stem,
        "route_name": log_entry.get("route_name") if log_entry else None,
    }


def split_by_route(
    video_stems: list[str],
    climb_log: dict[str, dict],
    train_fraction: float = 0.8,
    seed: int = 42,
) -> tuple[set[str], set[str]]:
    """
    Split video stems into train/test sets, grouping by route.

    Videos of the same route always go to the same split.
    Videos not in climb_log are assigned to training.

    Args:
        video_stems: All video stems in the dataset.
        climb_log: Dict from load_climb_log().
        train_fraction: Fraction of routes in training set.
        seed: Random seed for reproducibility.

    Returns:
        (train_stems, test_stems) as sets.
    """
    route_to_stems: dict[str, list[str]] = {}
    no_route = []

    for stem in video_stems:
        entry = climb_log.get(stem)
        route = entry.get("route_name") if entry else None
        if route:
            route_to_stems.setdefault(route, []).append(stem)
        else:
            no_route.append(stem)

    rng = np.random.default_rng(seed)
    routes = sorted(route_to_stems.keys())
    rng.shuffle(routes)

    n_train = max(1, int(len(routes) * train_fraction))
    train_routes = set(routes[:n_train])

    train_stems = set()
    test_stems = set()

    for route, stems in route_to_stems.items():
        target = train_stems if route in train_routes else test_stems
        target.update(stems)

    train_stems.update(no_route)

    return train_stems, test_stems


def build_dataset(
    poses_dir: Path | None = None,
    calibrations_dir: Path | None = None,
    train_fraction: float = 0.8,
    seed: int = 42,
    normalize_holds: bool = True,
    apply_edits: bool = True,
) -> dict:
    """
    Build the full training dataset from all paired videos.

    Args:
        poses_dir: Directory containing pose JSONs.
        calibrations_dir: Directory containing calibrations.
        train_fraction: Fraction of routes in training set.
        seed: Random seed for train/test split.
        normalize_holds: If True, normalize hold positions to [0, 1].
        apply_edits: If True, apply route edit exclusions. If False, use
            raw holds from the Kilter DB.

    Returns:
        Dict with:
            'train_sequences': list of (T_i, 17, 2) arrays
            'train_scores': list of (T_i, 17) arrays
            'train_holds': list of (N_i, 2) arrays, hold positions per video
            'train_roles': list of (N_i,) arrays, role IDs per video
            'test_sequences', 'test_scores', 'test_holds', 'test_roles': same for test
            'train_stems', 'test_stems': list of str
            'fps': float
            'metadata': dict with dataset statistics
    """
    paired = find_paired_videos(poses_dir, calibrations_dir)
    if not paired:
        raise RuntimeError(
            "No videos found with both poses and calibrations. "
            f"Searched {poses_dir or config.POSES_DIR} and "
            f"{calibrations_dir or config.CALIBRATIONS_DIR}"
        )

    climb_log = load_climb_log()

    processed = {}
    skipped = []
    no_route = []
    for pose_path, stem in paired:
        result = process_video(pose_path, stem, climb_log, normalize_holds, apply_edits)
        if result is not None:
            processed[stem] = result
        else:
            # Distinguish why it was skipped
            log_entry = climb_log.get(stem)
            if log_entry and resolve_route(log_entry) is None:
                no_route.append(stem)
            else:
                skipped.append(stem)

    if not processed:
        raise RuntimeError(
            "All videos had insufficient data or missing routes. "
            f"No route: {no_route}, Other: {skipped}"
        )

    train_stems, test_stems = split_by_route(
        list(processed.keys()), climb_log, train_fraction, seed
    )

    train_seqs, train_scores, train_holds, train_roles, train_names, train_route_holds = [], [], [], [], [], []
    test_seqs, test_scores, test_holds, test_roles, test_names, test_route_holds = [], [], [], [], [], []

    for stem, data in processed.items():
        if stem in train_stems:
            train_seqs.append(data["keypoints"])
            train_scores.append(data["scores"])
            train_holds.append(data["hold_positions"])
            train_roles.append(data["hold_roles"])
            train_route_holds.append(data["route_holds_raw"])
            train_names.append(stem)
        else:
            test_seqs.append(data["keypoints"])
            test_scores.append(data["scores"])
            test_holds.append(data["hold_positions"])
            test_roles.append(data["hold_roles"])
            test_route_holds.append(data["route_holds_raw"])
            test_names.append(stem)

    fps = next(iter(processed.values()))["fps"]
    total_train_frames = sum(s.shape[0] for s in train_seqs)
    total_test_frames = sum(s.shape[0] for s in test_seqs)

    return {
        "train_sequences": train_seqs,
        "train_scores": train_scores,
        "train_holds": train_holds,
        "train_roles": train_roles,
        "test_sequences": test_seqs,
        "test_scores": test_scores,
        "test_holds": test_holds,
        "test_roles": test_roles,
        "train_stems": train_names,
        "test_stems": test_names,
        "train_route_holds": train_route_holds,
        "test_route_holds": test_route_holds,
        "fps": fps,
        "metadata": {
            "n_train_videos": len(train_seqs),
            "n_test_videos": len(test_seqs),
            "n_skipped": len(skipped),
            "n_no_route": len(no_route),
            "skipped_stems": skipped,
            "no_route_stems": no_route,
            "total_train_frames": total_train_frames,
            "total_test_frames": total_test_frames,
            "train_fraction": train_fraction,
            "seed": seed,
            "holds_normalized": normalize_holds,
            "route_edits_applied": apply_edits,
        },
    }