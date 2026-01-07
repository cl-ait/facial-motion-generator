#!/usr/bin/env python3
"""
Facial Motion Generator Training Script
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.train import KoeMorphTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Facial Motion Generator model')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )
    parser.add_argument(
        '--prepare-cache',
        action='store_true',
        help='Only prepare cached features without training'
    )

    args = parser.parse_args()

    # Check if config exists
    if not Path(args.config).exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)

    # Prepare cache only mode
    if args.prepare_cache:
        print("=" * 60)
        print("Preparing cached features")
        print("=" * 60)

        from src.data.dataset import KoeMorphDataset
        import yaml

        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

        # Prepare train data
        print("\nPreparing training data...")
        train_dataset = KoeMorphDataset(
            data_root=config['data']['root'],
            split='train',
            cache_dir=config['data']['cache_dir'],
            use_cache=True
        )
        train_dataset.prepare_all_features()
        train_dataset.compute_statistics()

        # Prepare validation data
        print("\nPreparing validation data...")
        val_dataset = KoeMorphDataset(
            data_root=config['data']['root'],
            split='val',
            cache_dir=config['data']['cache_dir'],
            use_cache=True
        )
        val_dataset.prepare_all_features()

        # Prepare test data
        print("\nPreparing test data...")
        test_dataset = KoeMorphDataset(
            data_root=config['data']['root'],
            split='test',
            cache_dir=config['data']['cache_dir'],
            use_cache=True
        )
        test_dataset.prepare_all_features()

        print("\nCache preparation complete!")
        return

    # Start training
    trainer = KoeMorphTrainer(args.config)

    try:
        trainer.train(resume_from=args.resume)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint()
        print("Checkpoint saved. You can resume training with --resume flag")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
