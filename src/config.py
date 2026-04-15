from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import List, Optional
import json
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
    img_size: int = 224
    num_workers: int = 4

    # Model
    # HPO Run 2 best (trial 24, 2026-03-29): 512  |  previous (Run 1 best, used for training): 256
    embedding_dim: int = 512
    backbone: str = "resnet50"
    pretrained: bool = True

    # Training
    batch_size: int = 32
    samples_per_class: int = field(default=None)
    # HPO Run 2 best (trial 24, 2026-03-29): 2.808e-4  |  previous: 6.52e-4
    lr: float = 2.808e-4
    # HPO Run 2 best (trial 24, 2026-03-29): 5.372e-5  |  previous: 2.80e-4
    weight_decay: float = 5.372e-5
    epochs: int = 50
    patience: int = 7
    # HPO Run 2 best (trial 24, 2026-03-29): 0.3088  |  previous: 0.1826
    lr_backbone_factor: float = 0.3088  # LR multiplier for backbone (for fine-tuning)

    # Triplet Loss
    # HPO Run 2 best (trial 24, 2026-03-29): 0.1776  |  previous: 0.1484
    margin: float = 0.1776
    # HPO Run 2 best (trial 24, 2026-03-29): multi_similarity  |  previous: batch_hard
    mining_strategy: str = "multi_similarity"  # batch_hard, semihard, multi_similarity

    # CV
    n_folds: int = 5
    seed: int = 42

    # HPO
    hpo_n_trials: int = 30
    hpo_timeout: int = 25200  # seconds (7 hours)
    # 10 epochs per trial is a deliberate low-fidelity proxy: HPO needs to rank configurations,
    # not train them to convergence. Relative ordering of hyperparameter configurations stabilizes
    # early in training, and Optuna's MedianPruner already prunes underperforming trials at each
    # step. This follows the Hyperband/BOHB paradigm of using early stopping as a cheap
    # approximation, making it a principled choice rather than just a computational shortcut.
    hpo_epochs: int = 10
    # HPO subset sizes for embedding computation (validation only, not training)
    hpo_proto_samples_per_class: int = 100   # samples/class for prototype centroid fitting
    hpo_val_samples_per_class: int = 75     # samples/class for F1 scoring

    # Device
    device: str = field(default=None)
    use_amp: bool = False  # mixed-precision training (CUDA only; no-op on CPU)

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


def load_config_overrides(config: "Config") -> "Config":
    """Apply overrides from `<tables_dir>/config_override.json` if it exists.

    Used by Colab to swap in a fast runtime config (ResNet18, smaller images, AMP, fewer folds)
    without changing script code. Only fields that exist on Config are applied; unknown keys are
    warned about and ignored. Re-runs __post_init__-style derived fields that depend on overrides.
    """
    override_path = config.tables_dir / "config_override.json"
    if not override_path.exists():
        return config

    with open(override_path) as f:
        overrides = json.load(f)

    valid_names = {f.name for f in fields(config)}
    applied, unknown = {}, []
    for key, value in overrides.items():
        if key in valid_names:
            setattr(config, key, value)
            applied[key] = value
        else:
            unknown.append(key)

    # Recompute derived samples_per_class if batch_size changed and user didn't pin it
    if "batch_size" in applied and "samples_per_class" not in applied:
        config.samples_per_class = max(4, config.batch_size // config.num_classes)

    print(f"[config_override] Applied from {override_path.name}: {applied}")
    if unknown:
        print(f"[config_override] Ignored unknown keys: {unknown}")
    return config
