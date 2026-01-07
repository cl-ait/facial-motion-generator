"""
完全版音響特徴抽出器（log-mel + eGeMAPS）
"""

import numpy as np
import librosa
import opensmile
from collections import deque
from typing import Tuple, Optional
import threading
import time


class AudioFeatureExtractorComplete:
    """
    リアルタイム音響特徴抽出器
    - log-mel: 10ms更新、7フレームバッファ（短期特徴）
    - eGeMAPS: 300ms更新、5秒窓（長期特徴）
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_mels: int = 80,
        hop_length_ms: int = 10,
        win_length_ms: int = 25,
        egemaps_shift_ms: int = 300,
        egemaps_frame_ms: int = 5000
    ):
        self.sample_rate = sample_rate
        self.n_mels = n_mels

        # Log-mel parameters
        self.hop_length = int(sample_rate * hop_length_ms / 1000)  # 160 samples
        self.win_length = int(sample_rate * win_length_ms / 1000)  # 400 samples
        self.n_fft = self.win_length

        # eGeMAPS parameters
        self.egemaps_shift = egemaps_shift_ms / 1000  # 0.3 seconds
        self.egemaps_frame_samples = int(sample_rate * egemaps_frame_ms / 1000)  # 80000 samples

        # Initialize OpenSMILE
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals,
        )

        # Buffers
        self.mel_buffer = deque(maxlen=7)  # 7 frames of log-mel
        self.mel_audio_buffer = np.zeros(0, dtype=np.float32)
        self.egemaps_audio_buffer = np.zeros(self.egemaps_frame_samples, dtype=np.float32)
        self.egemaps_buffer_idx = 0
        self.egemaps_features = None

        # Timing
        self.last_egemaps_time = 0
        self.start_time = None

        # Statistics
        self.mel_stats = None
        self.egemaps_stats = None

        # Counters
        self.mel_frame_count = 0
        self.egemaps_update_count = 0

    def extract_log_mel(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Extract log-mel spectrogram from audio chunk"""
        if len(audio_chunk) < self.win_length:
            return None

        mel_spec = librosa.feature.melspectrogram(
            y=audio_chunk,
            sr=self.sample_rate,
            n_mels=self.n_mels,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            center=False
        )

        log_mel = np.log(mel_spec + 1e-10)

        # Apply normalization if available
        if self.mel_stats is not None:
            # Reshape stats for broadcasting: (80,) -> (80, 1)
            mean = self.mel_stats['mean'].reshape(-1, 1)
            std = self.mel_stats['std'].reshape(-1, 1)
            log_mel = (log_mel - mean) / (std + 1e-10)

        return log_mel

    def extract_egemaps(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """Extract eGeMAPS features from audio chunk"""
        try:
            features = self.smile.process_signal(audio_chunk, self.sample_rate)
            features_array = features.values.flatten()

            # Apply normalization if available
            if self.egemaps_stats is not None:
                # Ensure correct shapes for broadcasting
                mean = self.egemaps_stats['mean'].flatten()
                std = self.egemaps_stats['std'].flatten()
                features_array = (features_array - mean) / (std + 1e-10)

            return features_array
        except Exception as e:
            print(f"eGeMAPS extraction error: {e}")
            return None

    def update_egemaps_buffer(self, new_audio: np.ndarray):
        """Update circular buffer for eGeMAPS"""
        n_samples = len(new_audio)

        # Circular buffer update
        if self.egemaps_buffer_idx + n_samples <= self.egemaps_frame_samples:
            # Simple case: no wrap-around
            self.egemaps_audio_buffer[self.egemaps_buffer_idx:self.egemaps_buffer_idx + n_samples] = new_audio
            self.egemaps_buffer_idx += n_samples
        else:
            # Wrap-around case
            overflow = (self.egemaps_buffer_idx + n_samples) - self.egemaps_frame_samples
            self.egemaps_audio_buffer[self.egemaps_buffer_idx:] = new_audio[:-overflow]
            self.egemaps_audio_buffer[:overflow] = new_audio[-overflow:]
            self.egemaps_buffer_idx = overflow

    def process_audio(self, audio_chunk: np.ndarray, timestamp: Optional[float] = None) -> Tuple[bool, bool]:
        """
        Process audio chunk and extract features

        Returns:
            (mel_updated, egemaps_updated): Flags indicating which features were updated
        """
        if self.start_time is None:
            self.start_time = time.time()

        if timestamp is None:
            timestamp = time.time() - self.start_time

        mel_updated = False
        egemaps_updated = False

        # Update eGeMAPS buffer (always accumulate)
        self.update_egemaps_buffer(audio_chunk)

        # Process log-mel
        self.mel_audio_buffer = np.concatenate([self.mel_audio_buffer, audio_chunk])

        if len(self.mel_audio_buffer) >= self.win_length:
            log_mel = self.extract_log_mel(self.mel_audio_buffer)
            if log_mel is not None and log_mel.shape[1] > 0:
                # Add frames to buffer
                for frame_idx in range(log_mel.shape[1]):
                    self.mel_buffer.append(log_mel[:, frame_idx])
                    self.mel_frame_count += 1
                mel_updated = True

                # Remove processed audio
                consumed_samples = log_mel.shape[1] * self.hop_length
                self.mel_audio_buffer = self.mel_audio_buffer[consumed_samples:]

        # Process eGeMAPS (300ms update rate)
        if timestamp - self.last_egemaps_time >= self.egemaps_shift:
            egemaps = self.extract_egemaps(self.egemaps_audio_buffer)
            if egemaps is not None:
                self.egemaps_features = egemaps
                self.last_egemaps_time = timestamp
                self.egemaps_update_count += 1
                egemaps_updated = True

        return mel_updated, egemaps_updated

    def get_features(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get current features for inference

        Returns:
            (mel_features, egemaps_features):
                - mel_features: (80, 7) array or None
                - egemaps_features: (88,) array or None
        """
        # Get mel features
        if len(self.mel_buffer) == 7:
            mel_features = np.stack(list(self.mel_buffer), axis=1)  # (80, 7)
        else:
            mel_features = None

        # Get eGeMAPS features
        egemaps_features = self.egemaps_features

        return mel_features, egemaps_features

    def set_mel_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization statistics for mel features"""
        # Ensure numpy arrays
        self.mel_stats = {
            'mean': np.array(mean).flatten(),
            'std': np.array(std).flatten()
        }

    def set_egemaps_stats(self, mean: np.ndarray, std: np.ndarray):
        """Set normalization statistics for eGeMAPS features"""
        # Ensure numpy arrays
        self.egemaps_stats = {
            'mean': np.array(mean).flatten(),
            'std': np.array(std).flatten()
        }

    def get_status(self) -> dict:
        """Get current status of the feature extractor"""
        return {
            'mel_frames': self.mel_frame_count,
            'mel_buffer_size': len(self.mel_buffer),
            'egemaps_updates': self.egemaps_update_count,
            'egemaps_available': self.egemaps_features is not None,
            'ready_for_inference': len(self.mel_buffer) == 7 and self.egemaps_features is not None
        }
