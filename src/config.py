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
        "Tomato__Tomato_YellowLeaf__Curl_Virus",
        "Tomato__Tomato_mosaic_virus",
        "Tomato_healthy",
    ])
    unknown_class: str = "Tomato_Bacterial_spot"
    img_size: int = 80
    num_workers: int = 4

    # Model
    embedding_dim: int = 128
    backbone: str = "resnet18"
    pretrained: bool = True

    # Training
    batch_size: int = 32
    samples_per_class: int = field(default=None)
    lr: float = 6.52e-4
    weight_decay: float = 2.80e-4
    epochs: int = 30
    patience: int = 7
    lr_backbone_factor: float = 0.1826  # LR multiplier for backbone (for fine-tuning)
    freeze_backbone: bool = True  # Freeze backbone, train only embedding head

    # Triplet Loss
    margin: float = 0.1484
    mining_strategy: str = "batch_hard"  # batch_hard, semihard, multi_similarity

    # CV
    n_folds: int = 5
    seed: int = 42

    # HPO
    hpo_n_trials: int = 30
    hpo_timeout: int = 7200  # seconds
    hpo_epochs: int = 10  # epochs per trial (less than full training for speed)
    # HPO subset sizes for embedding computation (validation only, not training)
    hpo_proto_samples_per_class: int = 100   # samples/class for prototype centroid fitting
    hpo_val_samples_per_class: int = 75     # samples/class for F1 scoring

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
        
        if self.samples_per_class is None:
            self.samples_per_class = max(4, self.batch_size // self.num_classes)

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def get_known_classes(self) -> List[str]:
        """Classes for subtask b (excluding unknown_class)."""
        return [c for c in self.classes if c != self.unknown_class]
