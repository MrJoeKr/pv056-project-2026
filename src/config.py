from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import torch


@dataclass
class Config:
    # Paths
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(default=None)
    results_dir: Path = field(default=None)
    plots_dir: Path = field(default=None)
    models_dir: Path = field(default=None)
    logs_dir: Path = field(default=None)
    tables_dir: Path = field(default=None)

    # Dataset
    classes: List[str] = field(default_factory=lambda: [
        "Pepper__bell___Bacterial_spot",
        "Pepper__bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Tomato_Bacterial_spot",
        "Tomato_Early_blight",
        "Tomato_Late_blight",
        "Tomato_Leaf_Mold",
        "Tomato_Septoria_leaf_spot",
        "Tomato_Spider_mites_Two_spotted_spider_mite",
        "Tomato__Target_Spot",
    ])
    unknown_class: str = "Tomato_Bacterial_spot"
    img_size: int = 224
    num_workers: int = 4

    # Model
    embedding_dim: int = 512
    backbone: str = "resnet50"
    pretrained: bool = True

    # Training
    batch_size: int = 64
    samples_per_class: int = 4
    lr: float = 1e-4
    weight_decay: float = 1e-4
    epochs: int = 30
    patience: int = 7

    # Triplet Loss
    margin: float = 0.2
    mining_strategy: str = "batch_hard"  # batch_hard, semihard, multi_similarity

    # CV
    n_folds: int = 5
    seed: int = 42

    # HPO
    hpo_n_trials: int = 30
    hpo_timeout: int = 7200  # seconds

    # Device
    device: str = field(default=None)

    def __post_init__(self):
        if self.data_dir is None:
            self.data_dir = self.project_root / "data" / "PlantVillage"
        if self.results_dir is None:
            self.results_dir = self.project_root / "results"
        if self.plots_dir is None:
            self.plots_dir = self.results_dir / "plots"
        if self.models_dir is None:
            self.models_dir = self.results_dir / "models"
        if self.logs_dir is None:
            self.logs_dir = self.results_dir / "logs"
        if self.tables_dir is None:
            self.tables_dir = self.results_dir / "tables"
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_known_classes(self) -> List[str]:
        """Classes for subtask b (excluding unknown_class)."""
        return [c for c in self.classes if c != self.unknown_class]
