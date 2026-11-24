"""
Training utilities and model definitions are under src/.

The released checkpoints are the final source-independent experiment
checkpoints. This entry point deliberately does not retrain automatically.

For retraining, construct the datasets using the source-independent
manifests and use src.training.train_utils.train_one_epoch().
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

print("ESC-50 training package")
print("Training utilities:", ROOT / "src/training/train_utils.py")
print("Model architectures:", ROOT / "src/models/architectures.py")
print("Use the source-independent split manifests under data/.")
