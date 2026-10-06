"""Shared paths and validated experiment configuration."""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DUMMY_DATA_DIR = DATA_DIR / "dummy"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"
CHECKPOINTS_DIR = MODELS_DIR / "checkpoints"
OUTPUTS_DIR = ROOT_DIR / "outputs"
ACTIVE_DATA_DIR = RAW_DATA_DIR
CLASS_NAMES = ["segar", "tidak_segar", "busuk"]
NUM_CLASSES = 3
ANGLE_LABELS = ["atas", "serong_atas", "tengah", "serong_bawah", "bawah"]
TOMATO_IDS = ["T1", "T2"]
DUMMY_TOMATO_IDS = ["T1", "T2", "T3", "T4"]
DUMMY_NUM_DAYS = 10
NUM_DAYS_ESTIMATE = 14
IMAGE_SIZE = (128, 128)
IMAGE_CHANNELS = 3
BATCH_SIZE = 16
RANDOM_SEED = 42
EPOCHS = 50
LEARNING_RATE = 0.001
TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.7, 0.15, 0.15
FILENAME_PATTERN = (
    r"^(?P<tomato_id>T\d+)_(?P<day>D\d{2,3})_"
    r"(?P<angle>serong_atas|serong_bawah|atas|tengah|bawah)_"
    r"(?P<label>tidak_segar|segar|busuk)\.jpg$"
)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUTS_DIR / ".matplotlib"))
os.environ.setdefault("KERAS_HOME", str(MODELS_DIR / ".keras"))

@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 42
    image_size: int = 128
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 0.001
    dropout: float = 0.35
    weight_decay: float = 0.0001
    patience: int = 10
    train_samples_per_class: int = 300
    threads: int = 4
    confidence_threshold: float = 0.7
    architecture: str = "regularized"

    def __post_init__(self):
        for key in ("seed", "image_size", "batch_size", "epochs", "patience",
                    "train_samples_per_class", "threads"):
            value = getattr(self, key)
            if not isinstance(value, int) or value < (0 if key == "seed" else 1):
                raise ValueError(f"Invalid {key}: {value}")
        if not 32 <= self.image_size <= 512:
            raise ValueError("image_size must be 32..512")
        if not 0 < self.learning_rate < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid learning rate / dropout")
        if not 0 <= self.weight_decay < 1 or not 0 <= self.confidence_threshold <= 1:
            raise ValueError("Invalid regularization / threshold")
        if self.architecture not in {"baseline", "regularized"}:
            raise ValueError("CNN architecture must be baseline or regularized")

    @classmethod
    def load(cls, path=None):
        return cls(**json.loads(Path(path).read_text())) if path else cls()

    def to_dict(self):
        return asdict(self)

def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT_DIR / path
