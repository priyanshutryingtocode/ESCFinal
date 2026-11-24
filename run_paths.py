from pathlib import Path

ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

TRAIN_DIR = DATA_DIR / "train"
VALIDATION_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"

LEVEL1_CHECKPOINT = MODEL_DIR / "level1_vgg16_no_filter.pt"
LEVEL2_CHECKPOINT_DIR = MODEL_DIR / "level2"
