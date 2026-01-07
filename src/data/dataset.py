"""
KoeMorph学習用データセット
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import opensmile
from tqdm import tqdm
import pickle


class KoeMorphDataset(Dataset):
    """
    KoeMorph学習用データセット
    wav音声とブレンドシェイプのペアを扱う
    """

    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length_ms: int = 10,
        win_length_ms: int = 25,
        egemaps_shift_ms: int = 300,
        egemaps_frame_ms: int = 5000,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")

        # Audio parameters
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.n_fft = self.win_length
        self.n_mels = n_mels

        # eGeMAPS parameters
        self.egemaps_shift_ms = egemaps_shift_ms
        self.egemaps_frame_ms = egemaps_frame_ms
        self.egemaps_shift_frames = int(egemaps_shift_ms / hop_length_ms)  # 30 frames

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Load data paths and optional split manifest
        self.data_paths = self._load_data_paths()
        self.split_manifest = self._load_split_manifest()

        # Split data (8:1:1)
        self._split_data()

        # Compute or load statistics
        self.mel_stats = None
        self.egemaps_stats = None
        self.blendshape_stats = None

    def _load_data_paths(self) -> List[Path]:
        """Load all available data paths"""
        data_dir = self.data_root / "data" / "processed"
        if not data_dir.exists():
            data_dir = self.data_root / "data_sample"  # Fallback to sample

        paths = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        print(f"Found {len(paths)} recordings")
        return paths

    def _load_split_manifest(self) -> Optional[Dict[str, List[str]]]:
        manifest_path = self.data_root / "data" / "split_manifest.json"
        if manifest_path.exists():
            with open(manifest_path, 'r', encoding='utf-8') as f:
                try:
                    return json.load(f)
                except json.JSONDecodeError as exc:
                    print(f"Warning: Failed to parse split manifest: {exc}")
        return None

    def _split_data(self):
        """Split data into train/val/test (8:1:1)"""
        if self.split_manifest and self.split in self.split_manifest:
            name_to_path = {p.name: p for p in self.data_paths}
            selected = []
            for rec_id in self.split_manifest[self.split]:
                if rec_id in name_to_path:
                    selected.append(name_to_path[rec_id])
                else:
                    print(f"Warning: recording '{rec_id}' listed in manifest for split '{self.split}' not found.")
            self.data_paths = selected
            print(f"Split '{self.split}': {len(self.data_paths)} recordings (manifest)")
            return

        n_total = len(self.data_paths)
        n_test = n_total // 10
        n_val = n_total // 10
        n_train = n_total - n_test - n_val

        if self.split == 'train':
            self.data_paths = self.data_paths[:n_train]
        elif self.split == 'val':
            self.data_paths = self.data_paths[n_train:n_train+n_val]
        elif self.split == 'test':
            self.data_paths = self.data_paths[n_train+n_val:]

        print(f"Split '{self.split}': {len(self.data_paths)} recordings")

    def _extract_features_for_file(self, recording_path: Path) -> Dict:
        """Extract all features for a single recording"""
        cache_file = self.cache_dir / f"{recording_path.name}_{self.split}.pkl"

        # Load from cache if available
        if self.use_cache and cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)

        # Load audio
        audio_path = recording_path / "audio.wav"
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # Load blendshapes
        blendshapes_path = recording_path / "blendshapes.jsonl"
        blendshapes = []
        timestamps = []

        with open(blendshapes_path, 'r') as f:
            for line in f:
                data = json.loads(line)
                blendshapes.append(data['blendshapes'])
                timestamps.append(data['timestamp'])

        blendshapes = np.array(blendshapes, dtype=np.float32)
        timestamps = np.array(timestamps, dtype=np.float32)

        # Extract log-mel
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False
        )
        log_mel = np.log(mel_spec + 1e-10).T  # (T, 80)

        # Extract eGeMAPS (every 300ms with 5s window)
        egemaps_features = []
        egemaps_timestamps = []

        window_samples = int(self.sample_rate * self.egemaps_frame_ms / 1000)
        shift_samples = int(self.sample_rate * self.egemaps_shift_ms / 1000)

        for i in range(0, len(audio) - window_samples, shift_samples):
            window = audio[i:i+window_samples]
            features = self.smile.process_signal(window, self.sample_rate)
            egemaps_features.append(features.values.flatten())
            egemaps_timestamps.append(i / self.sample_rate)

        egemaps_features = np.array(egemaps_features, dtype=np.float32)
        egemaps_timestamps = np.array(egemaps_timestamps, dtype=np.float32)

        # Package data
        data = {
            'recording_id': recording_path.name,
            'audio': audio,
            'log_mel': log_mel,
            'egemaps': egemaps_features,
            'egemaps_timestamps': egemaps_timestamps,
            'blendshapes': blendshapes,
            'timestamps': timestamps
        }

        # Save to cache
        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

        return data

    def prepare_all_features(self):
        """Pre-extract features for all files (recommended before training)"""
        print(f"Extracting features for {self.split} set...")
        for path in tqdm(self.data_paths):
            self._extract_features_for_file(path)
        print("Feature extraction complete!")

    def compute_statistics(self):
        """Compute normalization statistics"""
        print("Computing normalization statistics...")

        all_mel = []
        all_egemaps = []
        all_blendshapes = []

        for path in tqdm(self.data_paths, desc="Loading features"):
            data = self._extract_features_for_file(path)
            all_mel.append(data['log_mel'])
            all_egemaps.append(data['egemaps'])
            all_blendshapes.append(data['blendshapes'])

        # Concatenate all features
        all_mel = np.concatenate(all_mel, axis=0)
        all_egemaps = np.concatenate(all_egemaps, axis=0)
        all_blendshapes = np.concatenate(all_blendshapes, axis=0)

        # Compute statistics
        self.mel_stats = {
            'mean': all_mel.mean(axis=0),
            'std': all_mel.std(axis=0) + 1e-10
        }

        self.egemaps_stats = {
            'mean': all_egemaps.mean(axis=0),
            'std': all_egemaps.std(axis=0) + 1e-10
        }

        self.blendshape_stats = {
            'mean': all_blendshapes.mean(axis=0),
            'std': all_blendshapes.std(axis=0) + 1e-10
        }

        print("Statistics computed!")

        # Save statistics
        stats_file = self.cache_dir / f"stats_{self.split}.pkl"
        with open(stats_file, 'wb') as f:
            pickle.dump({
                'mel': self.mel_stats,
                'egemaps': self.egemaps_stats,
                'blendshapes': self.blendshape_stats
            }, f)

    def load_statistics(self, stats_file: Optional[str] = None):
        """Load pre-computed statistics"""
        if stats_file is None:
            stats_file = self.cache_dir / f"stats_train.pkl"  # Use train stats

        with open(stats_file, 'rb') as f:
            stats = pickle.load(f)

        self.mel_stats = stats['mel']
        self.egemaps_stats = stats['egemaps']
        self.blendshape_stats = stats['blendshapes']

    def __len__(self):
        # For now, return number of recordings
        # In actual training, we'll sample windows from these
        return len(self.data_paths)

    def __getitem__(self, idx):
        """
        Get a training sample
        Returns normalized features ready for training
        """
        # This is a placeholder - actual implementation will sample windows
        data = self._extract_features_for_file(self.data_paths[idx])

        # Apply normalization if available
        if self.mel_stats:
            log_mel = (data['log_mel'] - self.mel_stats['mean']) / self.mel_stats['std']
        else:
            log_mel = data['log_mel']

        if self.egemaps_stats:
            egemaps = (data['egemaps'] - self.egemaps_stats['mean']) / self.egemaps_stats['std']
        else:
            egemaps = data['egemaps']

        if self.blendshape_stats:
            blendshapes = (data['blendshapes'] - self.blendshape_stats['mean']) / self.blendshape_stats['std']
        else:
            blendshapes = data['blendshapes']

        return {
            'recording_id': data['recording_id'],
            'log_mel': torch.FloatTensor(log_mel),
            'egemaps': torch.FloatTensor(egemaps),
            'blendshapes': torch.FloatTensor(blendshapes),
            'timestamps': torch.FloatTensor(data['timestamps'])
        }


class KoeMorphBatchSampler:
    """
    Custom batch sampler for temporal alignment
    Samples synchronized windows from recordings
    """

    def __init__(
        self,
        dataset: KoeMorphDataset,
        batch_size: int = 32,
        window_size: int = 100,  # frames
        mel_context: int = 7,  # past frames for mel
        skip_first_ms: int = 300  # Skip first 300ms (no eGeMAPS)
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.window_size = window_size
        self.mel_context = mel_context
        self.skip_frames = int(skip_first_ms / 10)  # 30 frames

        # Pre-compute valid sampling ranges for each recording
        self.valid_ranges = []
        for path in dataset.data_paths:
            data = dataset._extract_features_for_file(path)
            n_frames = len(data['timestamps'])

            # Valid range: after skip_frames and with enough context
            start = max(self.skip_frames, self.mel_context)
            end = n_frames - self.window_size

            if end > start:
                self.valid_ranges.append((path, start, end))

        self.n_samples = len(self.valid_ranges) * 10  # Sample 10 windows per recording

    def __iter__(self):
        # Generate random samples
        for _ in range(self.n_samples // self.batch_size):
            batch = []

            for _ in range(self.batch_size):
                # Random recording
                rec_idx = np.random.randint(len(self.valid_ranges))
                path, start, end = self.valid_ranges[rec_idx]

                # Random window position
                frame_idx = np.random.randint(start, end)

                batch.append((rec_idx, frame_idx))

            yield batch

    def __len__(self):
        return self.n_samples // self.batch_size
