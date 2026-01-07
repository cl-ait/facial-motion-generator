#!/usr/bin/env python3
"""
Facial Motion Generator Realtime Inference Script
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.realtime_inference import main

if __name__ == "__main__":
    main()
