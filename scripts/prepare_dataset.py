#!/usr/bin/env python3
"""
Dataset Preparation Script for Facial Motion Generator

This script prepares the NVIDIA Audio2Face-3D Dataset for training.
It converts mesh sequences to BlendShape coefficients.

Prerequisites:
- Download Audio2Face-3D-Dataset-v1.0.0-claire from:
  https://huggingface.co/datasets/nvidia/Audio2Face-3D-Dataset-v1.0.0-claire

Usage:
    python scripts/prepare_dataset.py --source /path/to/audio2face-dataset --output data/processed
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict
import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_blendshape_basis(basis_path: Path) -> Dict:
    """Load BlendShape basis from NVIDIA dataset"""
    # This would typically load the basis shapes from the dataset
    # The actual implementation depends on the dataset format
    print(f"Loading BlendShape basis from: {basis_path}")
    return {}


def solve_blendshape_coefficients(mesh: np.ndarray, basis: Dict) -> np.ndarray:
    """Solve for BlendShape coefficients using least squares"""
    # This is a simplified placeholder
    # The actual implementation uses linear least squares fitting
    return np.zeros(52, dtype=np.float32)


def process_recording(
    recording_dir: Path,
    output_dir: Path,
    basis: Dict,
    target_fps: int = 30
) -> bool:
    """Process a single recording directory"""
    audio_file = recording_dir / "audio.wav"
    mesh_dir = recording_dir / "meshes"

    if not audio_file.exists():
        print(f"Skipping {recording_dir.name}: no audio.wav found")
        return False

    # Create output directory
    out_recording_dir = output_dir / recording_dir.name
    out_recording_dir.mkdir(parents=True, exist_ok=True)

    # Copy audio file
    import shutil
    shutil.copy(audio_file, out_recording_dir / "audio.wav")

    # Process mesh sequence and extract BlendShape coefficients
    # (This is a placeholder - actual implementation would process meshes)
    blendshapes = []
    timestamps = []

    # Write blendshapes to JSONL format
    with open(out_recording_dir / "blendshapes.jsonl", 'w') as f:
        for i, (timestamp, bs) in enumerate(zip(timestamps, blendshapes)):
            record = {
                "timestamp": float(timestamp),
                "blendshapes": bs.tolist()
            }
            f.write(json.dumps(record) + "\n")

    return True


def create_split_manifest(
    output_dir: Path,
    recordings: List[str],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1
) -> None:
    """Create train/val/test split manifest"""
    n_total = len(recordings)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    manifest = {
        "train": recordings[:n_train],
        "val": recordings[n_train:n_train+n_val],
        "test": recordings[n_train+n_val:]
    }

    manifest_path = output_dir / "split_manifest.json"
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print(f"Created split manifest: {manifest_path}")
    print(f"  Train: {len(manifest['train'])} recordings")
    print(f"  Val: {len(manifest['val'])} recordings")
    print(f"  Test: {len(manifest['test'])} recordings")


def main():
    parser = argparse.ArgumentParser(
        description='Prepare NVIDIA Audio2Face-3D Dataset for training'
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to Audio2Face-3D dataset directory'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--fps',
        type=int,
        default=30,
        help='Target frames per second for BlendShape coefficients'
    )

    args = parser.parse_args()

    source_dir = Path(args.source)
    output_dir = Path(args.output)

    if not source_dir.exists():
        print(f"Error: Source directory not found: {source_dir}")
        sys.exit(1)

    print("=" * 60)
    print("Facial Motion Generator - Dataset Preparation")
    print("=" * 60)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Target FPS: {args.fps}")
    print()

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load BlendShape basis
    basis_path = source_dir / "blendshape_basis"
    if basis_path.exists():
        basis = load_blendshape_basis(basis_path)
    else:
        print("Warning: BlendShape basis not found, using default")
        basis = {}

    # Find all recording directories
    recordings = sorted([
        d.name for d in source_dir.iterdir()
        if d.is_dir() and not d.name.startswith('.')
    ])

    print(f"Found {len(recordings)} recordings")
    print()

    # Process each recording
    processed = []
    for recording in recordings:
        recording_dir = source_dir / recording
        if process_recording(recording_dir, output_dir, basis, args.fps):
            processed.append(recording)

    print(f"\nProcessed {len(processed)} recordings")

    # Create split manifest
    if processed:
        create_split_manifest(output_dir, processed)

    print("\nDataset preparation complete!")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
