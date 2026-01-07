"""
KoeMorph学習トレーナー
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import yaml
import time
from tqdm import tqdm
import numpy as np
from typing import Dict, Optional
import json
from src.train.losses import build_loss_fn

# Optional tensorboard import
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("Warning: TensorBoard not available. Install with: pip install tensorboard")


class KoeMorphTrainer:
    """
    学習管理クラス
    """

    def __init__(self, config_path: str):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.config['inference']['device'] == 'cuda'
            else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Paths
        self.checkpoint_dir = Path("checkpoints")
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)

        # Initialize components
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.train_loader = None
        self.val_loader = None
        self.writer = None

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.loss_fn = None
        self.metric_fn = nn.MSELoss()

    def setup_model(self):
        """Initialize model"""
        from src.models import KoeMorphCrossAttention

        model_config = self.config['model']
        self.model = KoeMorphCrossAttention(
            n_blendshapes=model_config['n_blendshapes'],
            embed_dim=model_config['embed_dim'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            mlp_hidden=model_config['mlp_hidden'],
            use_smoothing=False  # No smoothing during training
        ).to(self.device)

        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def setup_data(self):
        """Initialize data loaders"""
        from src.data.dataset_v2 import KoeMorphDatasetV2

        data_config = self.config['data']
        audio_config = self.config['audio']

        # Training dataset
        train_dataset = KoeMorphDatasetV2(
            data_root=data_config['root'],
            split='train',
            sample_rate=data_config['sample_rate'],
            n_mels=audio_config['mel']['n_mels'],
            hop_length_ms=audio_config['mel']['hop_length_ms'],
            win_length_ms=audio_config['mel']['win_length_ms'],
            egemaps_shift_ms=audio_config['egemaps']['shift_ms'],
            egemaps_frame_ms=audio_config['egemaps']['frame_ms'],
            mel_context=audio_config['mel']['context_frames'],
            cache_dir=data_config['cache_dir'],
            use_cache=True
        )

        # Load statistics
        train_dataset.load_statistics()

        # Validation dataset
        val_dataset = KoeMorphDatasetV2(
            data_root=data_config['root'],
            split='val',
            sample_rate=data_config['sample_rate'],
            n_mels=audio_config['mel']['n_mels'],
            hop_length_ms=audio_config['mel']['hop_length_ms'],
            win_length_ms=audio_config['mel']['win_length_ms'],
            egemaps_shift_ms=audio_config['egemaps']['shift_ms'],
            egemaps_frame_ms=audio_config['egemaps']['frame_ms'],
            mel_context=audio_config['mel']['context_frames'],
            cache_dir=data_config['cache_dir'],
            use_cache=True
        )

        # Use training statistics for validation
        val_dataset.load_statistics()

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=True  # Drop incomplete batches
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=2,
            pin_memory=True if self.device.type == 'cuda' else False,
            drop_last=False  # Use all data for validation
        )

        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")

    def setup_optimization(self):
        """Initialize optimizer and scheduler"""
        train_config = self.config['training']

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=train_config['learning_rate'],
            weight_decay=train_config['weight_decay']
        )
        self.loss_fn = build_loss_fn(train_config['loss'])

        # Scheduler
        if train_config['scheduler']['type'] == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=train_config['num_epochs'],
                eta_min=train_config['scheduler']['min_lr']
            )

    def setup_logging(self):
        """Initialize logging"""
        if self.config['logging']['tensorboard'] and TENSORBOARD_AVAILABLE:
            self.writer = SummaryWriter(self.log_dir / f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        elif self.config['logging']['tensorboard'] and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not available")
            self.writer = None

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()

        total_loss = 0
        n_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch in pbar:
            # Move to device
            mel = batch['mel'].to(self.device)
            egemaps = batch['egemaps'].to(self.device)
            target = batch['blendshape'].to(self.device)

            # Forward pass
            pred = self.model(mel, egemaps)

            # Compute loss
            loss = self.loss_fn(pred, target)

            # Add smoothness penalty if configured
            if self.config['training']['loss']['weights'].get('smoothness', 0) > 0:
                smoothness_weight = self.config['training']['loss']['weights']['smoothness']
                if n_batches > 0 and hasattr(self, 'prev_pred'):
                    # Only compute smoothness if batch sizes match
                    if pred.shape[0] == self.prev_pred.shape[0]:
                        smoothness_loss = nn.MSELoss()(pred, self.prev_pred.detach())
                        loss = loss + smoothness_weight * smoothness_loss
                self.prev_pred = pred.clone()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.config['training']['gradient_clip'] > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # Update statistics
            total_loss += loss.item()
            n_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

            # Log to tensorboard
            if self.writer and self.global_step % self.config['logging']['log_every'] == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/mse', self.metric_fn(pred, target).item(), self.global_step)
                self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        return total_loss / n_batches

    def validate(self):
        """Validate model"""
        self.model.eval()

        total_loss = 0
        total_mse = 0
        n_batches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Move to device
                mel = batch['mel'].to(self.device)
                egemaps = batch['egemaps'].to(self.device)
                target = batch['blendshape'].to(self.device)

                # Forward pass
                pred = self.model(mel, egemaps)

                # Handle different batch sizes (last batch might be smaller)
                if pred.shape[0] != target.shape[0]:
                    min_batch = min(pred.shape[0], target.shape[0])
                    pred = pred[:min_batch]
                    target = target[:min_batch]

                # Compute loss
                loss = self.loss_fn(pred, target)
                mse_metric = self.metric_fn(pred, target)

                total_loss += loss.item()
                total_mse += mse_metric.item()
                n_batches += 1

        val_loss = total_loss / n_batches
        val_mse = total_mse / n_batches

        # Log to tensorboard
        if self.writer:
            self.writer.add_scalar('val/loss', val_loss, self.global_step)
            self.writer.add_scalar('val/mse', val_mse, self.global_step)

        return val_loss

    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }

        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{self.current_epoch}.pth"
        torch.save(checkpoint, checkpoint_path)

        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model with val_loss: {self.best_val_loss:.4f}")

        # Keep only last N checkpoints
        keep_last = self.config['training']['checkpoint']['keep_last']
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_epoch_*.pth"))
        if len(checkpoints) > keep_last:
            for old_checkpoint in checkpoints[:-keep_last]:
                old_checkpoint.unlink()

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Loaded checkpoint from epoch {self.current_epoch}")

    def train(self, resume_from: Optional[str] = None):
        """Main training loop"""
        # Setup
        self.setup_model()
        self.setup_data()
        self.setup_optimization()
        self.setup_logging()

        # Resume from checkpoint
        if resume_from:
            self.load_checkpoint(resume_from)

        # Training loop
        num_epochs = self.config['training']['num_epochs']

        print("\n" + "=" * 60)
        print("Starting training")
        print("=" * 60)

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss = self.validate()

            # Update scheduler
            if self.scheduler:
                self.scheduler.step()

            # Check if best model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss

            # Save checkpoint
            if (epoch + 1) % self.config['training']['checkpoint']['save_every'] == 0:
                self.save_checkpoint(is_best)

            # Print epoch summary
            print(f"\nEpoch {epoch}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Best Val Loss: {self.best_val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")

        # Close writer
        if self.writer:
            self.writer.close()

        print("\n" + "=" * 60)
        print("Training completed!")
        print("=" * 60)
