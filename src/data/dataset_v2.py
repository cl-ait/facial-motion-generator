"""
改良版KoeMorphデータセット（ウィンドウサンプリング対応）
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import librosa
import opensmile
from tqdm import tqdm
import pickle
import random


class KoeMorphDatasetV2(Dataset):
    """
    ウィンドウサンプリング対応のデータセット
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
        mel_context: int = 7,
        skip_first_ms: int = 300,
        samples_per_recording: int = 100,
        cache_dir: Optional[str] = None,
        use_cache: bool = True
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.sample_rate = sample_rate
        self.use_cache = use_cache
        self.cache_dir = Path(cache_dir) if cache_dir else Path("cache")

        # Window sampling parameters
        self.mel_context = mel_context
        self.skip_frames = int(skip_first_ms / hop_length_ms)
        self.samples_per_recording = samples_per_recording

        # Audio parameters
        self.hop_length = int(sample_rate * hop_length_ms / 1000)
        self.win_length = int(sample_rate * win_length_ms / 1000)
        self.n_fft = self.win_length
        self.n_mels = n_mels
        self.hop_length_ms = hop_length_ms

        # eGeMAPS parameters
        self.egemaps_shift_ms = egemaps_shift_ms
        self.egemaps_frame_ms = egemaps_frame_ms

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Load data paths and optional split manifest
        self.data_paths = self._load_data_paths()
        self.split_manifest = self._load_split_manifest()
        self._split_data()

        # Precompute all features
        self.all_features = []
        self._load_all_features()

        # Compute valid windows for each recording
        self.valid_windows = []
        self._compute_valid_windows()

        # Statistics
        self.mel_stats = None
        self.egemaps_stats = None
        self.blendshape_stats = None

    def _load_data_paths(self) -> List[Path]:
        """Load all available data paths"""
        data_dir = self.data_root / "data" / "processed"
        if not data_dir.exists():
            data_dir = self.data_root / "data_sample"

        paths = sorted([p for p in data_dir.iterdir() if p.is_dir()])
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

    def _extract_features_for_file(self, recording_path: Path) -> Dict:
        """Extract all features for a single recording"""
        cache_file = self.cache_dir / f"{recording_path.name}_{self.split}.pkl"

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
        log_mel = np.log(mel_spec + 1e-10).T

        # Extract eGeMAPS
        egemaps_features = []
        egemaps_timestamps = []

        window_samples = int(self.sample_rate * self.egemaps_frame_ms / 1000)
        shift_samples = int(self.sample_rate * self.egemaps_shift_ms / 1000)

        for i in range(0, len(audio) - window_samples + 1, shift_samples):
            window = audio[i:i+window_samples]
            features = self.smile.process_signal(window, self.sample_rate)
            egemaps_features.append(features.values.flatten())
            egemaps_timestamps.append(i / self.sample_rate)

        egemaps_features = np.array(egemaps_features, dtype=np.float32)
        egemaps_timestamps = np.array(egemaps_timestamps, dtype=np.float32)

        data = {
            'recording_id': recording_path.name,
            'log_mel': log_mel,
            'egemaps': egemaps_features,
            'egemaps_timestamps': egemaps_timestamps,
            'blendshapes': blendshapes,
            'timestamps': timestamps
        }

        if self.use_cache:
            self.cache_dir.mkdir(exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)

        return data

    def _load_all_features(self):
        """Load all features into memory"""
        print(f"Loading {self.split} features...")
        for path in tqdm(self.data_paths):
            features = self._extract_features_for_file(path)
            self.all_features.append(features)

    def _compute_valid_windows(self):
        """Compute valid sampling windows for each recording"""
        for features in self.all_features:
            n_blendshapes = len(features['timestamps'])

            # Find valid range (after skip_frames and with enough context)
            for i in range(n_blendshapes):
                timestamp = features['timestamps'][i]
                mel_frame_idx = int(timestamp * 1000 / self.hop_length_ms)

                # Check if we have enough mel context
                if mel_frame_idx < self.mel_context:
                    continue

                # Check if we're past the skip period
                if timestamp < self.skip_frames * self.hop_length_ms / 1000:
                    continue

                # Check if mel frame exists
                if mel_frame_idx >= len(features['log_mel']):
                    continue

                self.valid_windows.append({
                    'features': features,
                    'blendshape_idx': i,
                    'timestamp': timestamp
                })

    def load_statistics(self, stats_file: Optional[str] = None):
        """Load pre-computed statistics"""
        if stats_file is None:
            stats_file = self.cache_dir / "stats_train.pkl"

        if Path(stats_file).exists():
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)

            self.mel_stats = stats['mel']
            self.egemaps_stats = stats['egemaps']
            self.blendshape_stats = stats.get('blendshapes', None)

    def __len__(self):
        return min(len(self.valid_windows),
                   len(self.all_features) * self.samples_per_recording)

    def __getitem__(self, idx):
        """Get a training sample"""
        # Random sample from valid windows
        window_info = random.choice(self.valid_windows)
        features = window_info['features']
        blendshape_idx = window_info['blendshape_idx']
        timestamp = window_info['timestamp']

        # Get target blendshape
        target_blendshape = features['blendshapes'][blendshape_idx]

        # Get mel context (past 7 frames)
        mel_frame_idx = int(timestamp * 1000 / self.hop_length_ms)
        mel_start = mel_frame_idx - self.mel_context + 1
        mel_end = mel_frame_idx + 1
        mel_window = features['log_mel'][mel_start:mel_end].T  # (80, 7)

        # Get corresponding eGeMAPS
        egemaps_idx = np.searchsorted(features['egemaps_timestamps'], timestamp)
        if egemaps_idx > 0:
            egemaps_idx -= 1
        egemaps_window = features['egemaps'][min(egemaps_idx, len(features['egemaps'])-1)]

        # Apply normalization
        if self.mel_stats:
            mel_window = (mel_window - self.mel_stats['mean'][:, None]) / self.mel_stats['std'][:, None]

        if self.egemaps_stats:
            egemaps_window = (egemaps_window - self.egemaps_stats['mean']) / self.egemaps_stats['std']

        return {
            'mel': torch.FloatTensor(mel_window),  # (80, 7)
            'egemaps': torch.FloatTensor(egemaps_window),  # (88,)
            'blendshape': torch.FloatTensor(target_blendshape),  # (52,)
            'timestamp': timestamp,
            'recording_id': features['recording_id']
        }
