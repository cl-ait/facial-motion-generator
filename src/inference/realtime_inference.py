"""
KoeMorphリアルタイム推論システム
マイク入力から60Hzでブレンドシェイプを推定
"""

import torch
import sounddevice as sd
import numpy as np
import time
import threading
import queue
from pathlib import Path
import yaml
from typing import Optional, Dict
from pythonosc import udp_client
import argparse
import sys

from src.models import KoeMorphCrossAttention
from src.inference.audio_features import AudioFeatureExtractorComplete
from src.utils.calibration import maybe_load_calibrator


class RealtimeInference:
    """
    リアルタイム推論システム
    """

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = "configs/default.yaml",
        device: Optional[str] = None,
        osc_host: str = "127.0.0.1",
        osc_port: int = 9000,
        verbose: bool = False
    ):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Device setup
        if device is None:
            device = self.config['inference']['device']

        # Debug information
        print(f"Device requested: {device}")
        print(f"CUDA available: {torch.cuda.is_available()}")

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        )
        print(f"Using device: {self.device}")

        # Audio parameters
        self.sample_rate = self.config['data']['sample_rate']
        self.block_size = 1600  # 100ms at 16kHz
        self.verbose = verbose

        # Load model
        self.model = self._load_model(checkpoint_path)

        # Feature extractor
        self.feature_extractor = AudioFeatureExtractorComplete(
            sample_rate=self.sample_rate,
            n_mels=self.config['audio']['mel']['n_mels'],
            hop_length_ms=self.config['audio']['mel']['hop_length_ms'],
            win_length_ms=self.config['audio']['mel']['win_length_ms'],
            egemaps_shift_ms=self.config['audio']['egemaps']['shift_ms'],
            egemaps_frame_ms=self.config['audio']['egemaps']['frame_ms']
        )

        # Load normalization statistics
        self._load_statistics()

        # Optional calibration
        calib_path = self.config["inference"].get("calibration_path")
        self.calibrator = maybe_load_calibrator(Path(calib_path) if calib_path else None)
        if self.calibrator:
            print(f"Loaded calibration from {calib_path}")

        # OSC client
        self.osc_client = udp_client.SimpleUDPClient(osc_host, osc_port)
        print(f"OSC output: {osc_host}:{osc_port}")

        # Threading components
        self.audio_queue = queue.Queue()
        self.inference_queue = queue.Queue(maxsize=2)
        self.is_running = False

        # Timing
        self.start_time = None
        self.last_inference_time = 0
        self.inference_interval = 1.0 / self.config['inference']['update_rate']  # 60Hz

        # Statistics
        self.inference_count = 0
        self.audio_frame_count = 0

    def _load_model(self, checkpoint_path: str) -> KoeMorphCrossAttention:
        """Load trained model from checkpoint"""
        print(f"Loading model from: {checkpoint_path}")

        # Create model
        model_config = self.config['model']
        model = KoeMorphCrossAttention(
            n_blendshapes=model_config['n_blendshapes'],
            embed_dim=model_config['embed_dim'],
            d_model=model_config['d_model'],
            n_heads=model_config['n_heads'],
            mlp_hidden=model_config['mlp_hidden'],
            use_smoothing=model_config['use_smoothing'],
            smoothing_window=model_config['smoothing_window']
        ).to(self.device)

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        # Set to evaluation mode
        model.eval()

        # Freeze base embeddings for inference
        model.freeze_base_embeddings()

        print(f"Model loaded successfully")
        return model

    def _load_statistics(self):
        """Load normalization statistics"""
        stats_file = Path(self.config['data']['cache_dir']) / "stats_train.pkl"

        if stats_file.exists():
            import pickle
            with open(stats_file, 'rb') as f:
                stats = pickle.load(f)

            # Convert to numpy arrays and set statistics
            import numpy as np
            self.feature_extractor.set_mel_stats(
                np.array(stats['mel']['mean']),
                np.array(stats['mel']['std'])
            )
            self.feature_extractor.set_egemaps_stats(
                np.array(stats['egemaps']['mean']),
                np.array(stats['egemaps']['std'])
            )
            print("Normalization statistics loaded")
        else:
            print("Warning: No normalization statistics found, using raw features")

    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status and self.verbose:
            print(f"Audio callback status: {status}")

        # Extract mono audio
        audio_data = indata[:, 0].copy().astype(np.float32)

        # Add to queue
        if self.start_time is None:
            self.start_time = time.time()

        timestamp = time.time() - self.start_time
        self.audio_queue.put((audio_data, timestamp))
        self.audio_frame_count += 1

    def audio_processing_thread(self):
        """Process audio and extract features"""
        print("Audio processing thread started")

        while self.is_running:
            try:
                # Get audio from queue
                audio_data, timestamp = self.audio_queue.get(timeout=0.1)

                # Extract features
                mel_updated, egemaps_updated = self.feature_extractor.process_audio(
                    audio_data, timestamp
                )

                # Get current features
                mel_features, egemaps_features = self.feature_extractor.get_features()

                # Check if ready for inference
                if mel_features is not None and egemaps_features is not None:
                    # Add to inference queue (non-blocking)
                    try:
                        self.inference_queue.put_nowait({
                            'mel': mel_features,
                            'egemaps': egemaps_features,
                            'timestamp': timestamp
                        })
                    except queue.Full:
                        pass  # Skip if queue is full

            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in audio processing: {e}")

    def inference_thread(self):
        """Run inference at 60Hz"""
        print("Inference thread started")

        last_features = None

        while self.is_running:
            current_time = time.time()

            # Check if it's time for inference
            if current_time - self.last_inference_time >= self.inference_interval:
                # Get latest features (non-blocking)
                try:
                    last_features = self.inference_queue.get_nowait()
                except queue.Empty:
                    pass

                # Run inference if features available
                if last_features is not None:
                    try:
                        # Prepare tensors
                        mel_tensor = torch.FloatTensor(last_features['mel']).unsqueeze(0).to(self.device)
                        egemaps_tensor = torch.FloatTensor(last_features['egemaps']).unsqueeze(0).to(self.device)

                        # Run inference
                        with torch.no_grad():
                            blendshapes = self.model(mel_tensor, egemaps_tensor)

                        # Convert to numpy
                        blendshapes_np = blendshapes.cpu().numpy()[0]
                        if self.calibrator is not None:
                            blendshapes_np = self.calibrator.apply(blendshapes_np)

                        # Send via OSC
                        self.send_blendshapes(blendshapes_np)

                        self.inference_count += 1

                        # Print status
                        if self.verbose and self.inference_count % 60 == 0:
                            elapsed = current_time - self.start_time
                            fps = self.inference_count / elapsed
                            print(f"[{elapsed:.1f}s] Inference: {self.inference_count}, FPS: {fps:.1f}")

                    except Exception as e:
                        print(f"Error in inference: {e}")

                self.last_inference_time = current_time
            else:
                # Sleep for a short time
                time.sleep(0.001)

    def send_blendshapes(self, blendshapes: np.ndarray):
        """Send blendshapes via OSC"""
        # Send as array
        self.osc_client.send_message("/blendshapes", blendshapes.tolist())

        # Also send individual values (for compatibility)
        for i, value in enumerate(blendshapes):
            self.osc_client.send_message(f"/blendshape/{i}", float(value))

    def run(self):
        """Main execution loop"""
        print("\n" + "=" * 60)
        print("Facial Motion Generator - Realtime Inference")
        print("=" * 60)
        print(f"Sample rate: {self.sample_rate} Hz")
        print(f"Inference rate: {self.config['inference']['update_rate']} Hz")
        print("\nWaiting for audio input...")
        print("(First 5 seconds for eGeMAPS buffer initialization)")
        print("\nPress Ctrl+C to stop")
        print("-" * 60)

        self.is_running = True

        # Start threads
        audio_thread = threading.Thread(target=self.audio_processing_thread)
        audio_thread.start()

        inference_thread_obj = threading.Thread(target=self.inference_thread)
        inference_thread_obj.start()

        try:
            # Start audio stream
            with sd.InputStream(
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                callback=self.audio_callback
            ):
                # Keep running
                while self.is_running:
                    time.sleep(1)

                    # Print periodic status
                    if self.start_time:
                        elapsed = time.time() - self.start_time
                        if int(elapsed) % 10 == 0 and self.verbose:
                            status = self.feature_extractor.get_status()
                            print(f"\n[{elapsed:.0f}s] Status:")
                            print(f"  Audio frames: {self.audio_frame_count}")
                            print(f"  Mel frames: {status['mel_frames']}")
                            print(f"  eGeMAPS updates: {status['egemaps_updates']}")
                            print(f"  Inferences: {self.inference_count}")
                            print(f"  Ready: {status['ready_for_inference']}")

        except KeyboardInterrupt:
            print("\n\nStopping...")
        except Exception as e:
            print(f"\nError: {e}")
        finally:
            self.is_running = False
            audio_thread.join()
            inference_thread_obj.join()

            # Print final statistics
            if self.start_time:
                elapsed = time.time() - self.start_time
                print("\n" + "=" * 60)
                print("Session Statistics")
                print("=" * 60)
                print(f"Duration: {elapsed:.1f} seconds")
                print(f"Audio frames: {self.audio_frame_count}")
                print(f"Inferences: {self.inference_count}")
                print(f"Average FPS: {self.inference_count / elapsed:.1f}")

                status = self.feature_extractor.get_status()
                print(f"Mel frames extracted: {status['mel_frames']}")
                print(f"eGeMAPS updates: {status['egemaps_updates']}")


def main():
    parser = argparse.ArgumentParser(description='Facial Motion Generator - Realtime Inference')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pth',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/default.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default=None,
        help='Device to use (cpu/cuda)'
    )
    parser.add_argument(
        '--osc-host',
        type=str,
        default='127.0.0.1',
        help='OSC host address'
    )
    parser.add_argument(
        '--osc-port',
        type=int,
        default=9000,
        help='OSC port'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose output'
    )

    args = parser.parse_args()

    # Check if checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Create inference system
    inference = RealtimeInference(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        osc_host=args.osc_host,
        osc_port=args.osc_port,
        verbose=args.verbose
    )

    # Run
    inference.run()


if __name__ == "__main__":
    main()
