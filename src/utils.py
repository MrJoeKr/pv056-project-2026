import random
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(device_str: str = None) -> torch.device:
    """Get torch device."""
    if device_str:
        return torch.device(device_str)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_checkpoint(model, optimizer, epoch, fold, metrics, path: Path):
    """Save model checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": epoch,
        "fold": fold,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }, path)


def load_checkpoint(path: Path, model, optimizer=None, device="cpu"):
    """Load model checkpoint."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


@torch.no_grad()
def compute_all_embeddings(model, dataloader, device):
    """Compute embeddings for all samples in a dataloader.

    Returns:
        embeddings: Tensor of shape (N, embedding_dim)
        labels: Tensor of shape (N,)
    """
    model.eval()
    all_embeddings = []
    all_labels = []
    for images, targets in tqdm(dataloader, desc="Computing embeddings", leave=False):
        images = images.to(device)
        embeddings = model(images)
        all_embeddings.append(embeddings.cpu())
        all_labels.append(targets)
    return torch.cat(all_embeddings), torch.cat(all_labels)
