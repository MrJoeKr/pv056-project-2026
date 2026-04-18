# ============================================================================
# Convolutional Autoencoder (ConvAE) for Outlier Detection
# ============================================================================


from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ConvolutionalAutoencoder(torch.nn.Module):
    """Vanilla Convolutional Autoencoder with symmetric encoder/decoder.
    
    Architecture:
        Encoder: 2-4 Conv layers with pooling
        Decoder: symmetric upsampling + transposed convolutions
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 64,
        hidden_dims: Optional[List[int]] = None,
    ):
        """Initialize ConvAE.
        
        Args:
            in_channels: Number of input channels (default: 3 for RGB)
            img_size: Image size (64x64 or 96x96)
            hidden_dims: List of hidden channel dimensions for each conv layer.
                        Default: [32, 64, 128, 256] for 4-layer encoder
        """
        super().__init__()
        self.in_channels = in_channels
        self.img_size = img_size
        self.hidden_dims = hidden_dims or [32, 64, 128, 256]
        
        # ---- ENCODER ----
        encoder_layers = []
        input_channels = in_channels
        for hidden_dim in self.hidden_dims:
            encoder_layers.append(
                torch.nn.Sequential(
                    torch.nn.Conv2d(
                        input_channels, hidden_dim, kernel_size=3, stride=1, padding=1
                    ),
                    torch.nn.ReLU(),
                    torch.nn.MaxPool2d(kernel_size=2, stride=2),
                )
            )
            input_channels = hidden_dim
        
        self.encoder = torch.nn.Sequential(*encoder_layers)
        
        # Compute bottleneck spatial dimensions
        # After each MaxPool2d with stride=2, spatial dims are halved
        self.bottleneck_spatial = img_size // (2 ** len(self.hidden_dims))
        self.bottleneck_channels = self.hidden_dims[-1]
        self.bottleneck_size = (
            self.bottleneck_channels,
            self.bottleneck_spatial,
            self.bottleneck_spatial,
        )
        
        # ---- DECODER (symmetric) ----
        decoder_layers = []
        hidden_dims_reversed = list(reversed(self.hidden_dims))
        
        for i in range(len(hidden_dims_reversed) - 1):
            decoder_layers.extend([
                torch.nn.ConvTranspose2d(
                    hidden_dims_reversed[i],
                    hidden_dims_reversed[i + 1],
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                ),
                torch.nn.ReLU(),
            ])
        
        # Final layer: reconstruct to input channels
        decoder_layers.append(
            torch.nn.ConvTranspose2d(
                hidden_dims_reversed[-1],
                in_channels,
                kernel_size=3,
                stride=2,
                padding=1,
                output_padding=1,
            )
        )
        decoder_layers.append(torch.nn.Sigmoid())  # Output in [0, 1]
        
        self.decoder = torch.nn.Sequential(*decoder_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: encode and decode."""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to bottleneck representation."""
        return self.encoder(x)


class ImageDataset(Dataset):
    """Simple dataset for image reconstruction."""
    
    def __init__(self, image_paths: List[str], img_size: int = 64):
        """Initialize dataset.
        
        Args:
            image_paths: List of paths to image files
            img_size: Target size for resizing (img_size x img_size)
        """
        self.image_paths = image_paths
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Load and preprocess image."""
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert("RGB")
            # Resize to target size
            img = img.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
            # Convert to tensor and normalize to [0, 1]
            img_array = np.array(img, dtype=np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)
            return img_tensor
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank tensor as fallback
            return torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)


def save_convautoencoder_checkpoint(
    model: ConvolutionalAutoencoder,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    losses: List[float],
    path: Path,
) -> None:
    """Save ConvAE checkpoint with model, optimizer, and training state.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch (0-indexed)
        losses: List of losses up to current epoch
        path: Path to save checkpoint
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "losses": losses,
        "img_size": model.img_size,
        "hidden_dims": model.hidden_dims,
        "in_channels": model.in_channels,
    }
    torch.save(checkpoint_dict, path)


def load_convautoencoder_checkpoint(
    path: Path,
    model: ConvolutionalAutoencoder,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Union[str, torch.device] = "cpu",
) -> Dict[str, Any]:
    """Load ConvAE checkpoint and restore training state.
    
    Args:
        path: Path to checkpoint
        model: Model to load into
        optimizer: Optimizer to load state into (optional)
        device: Device to load onto
        
    Returns:
        Checkpoint dict with epoch and losses
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    
    return {
        "epoch": checkpoint["epoch"],
        "losses": checkpoint["losses"],
    }


def train_convautoencoder(
    image_paths: List[str],
    img_size: int = 64,
    batch_size: int = 64,
    epochs: int = 20,
    learning_rate: float = 1e-3,
    device: Optional[torch.device] = None,
    hidden_dims: Optional[List[int]] = None,
    checkpoint_dir: Optional[Path] = None,
    save_interval: int = 5,
    resume_from: Optional[Path] = None,
) -> Tuple[ConvolutionalAutoencoder, List[float]]:
    """Train Convolutional Autoencoder on images with checkpointing.
    
    Args:
        image_paths: List of image file paths for training
        img_size: Target image size (64 or 96)
        batch_size: Batch size (default: 64)
        epochs: Number of training epochs (default: 20)
        learning_rate: Adam learning rate
        device: PyTorch device (cuda or cpu)
        hidden_dims: Channel dimensions for encoder layers
        checkpoint_dir: Directory to save checkpoints (optional)
        save_interval: Save checkpoint every N epochs (default: 5)
        resume_from: Path to checkpoint to resume from (optional)
    
    Returns:
        Trained model and list of epoch losses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create dataset and loader
    dataset = ImageDataset(image_paths, img_size=img_size)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )
    
    # Initialize model
    model = ConvolutionalAutoencoder(
        in_channels=3,
        img_size=img_size,
        hidden_dims=hidden_dims,
    ).to(device)
    
    # Loss and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    start_epoch = 0
    losses = []
    best_loss = float("inf")
    
    # Resume from checkpoint if provided
    if resume_from is not None and resume_from.exists():
        print(f"Resuming from checkpoint: {resume_from}")
        ckpt = load_convautoencoder_checkpoint(resume_from, model, optimizer, device)
        start_epoch = ckpt["epoch"] + 1
        losses = ckpt["losses"]
        best_loss = min(losses) if losses else float("inf")
        print(f"  Resumed at epoch {start_epoch}, best loss so far: {best_loss:.6f}")
    
    print(f"Training ConvAE on {len(dataset)} images for {epochs} epochs...")
    print(f"  Device: {device}")
    print(f"  Image size: {model.img_size}x{model.img_size}")
    print(f"  Hidden dims: {model.hidden_dims}")
    if checkpoint_dir:
        print(f"  Checkpoints: {checkpoint_dir}")
    
    for epoch in tqdm(range(start_epoch, epochs), desc="Epochs", unit="epoch"):
        epoch_loss = 0.0
        n_batches = 0
        
        for x in tqdm(loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False):
            x = x.to(device)
            
            # Forward pass
            x_recon = model(x)
            loss = criterion(x_recon, x)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        losses.append(avg_loss)
        
        # Track best loss
        if avg_loss < best_loss:
            best_loss = avg_loss
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{epochs}: MSE loss = {avg_loss:.6f}")
        
        # Save periodic checkpoint
        if checkpoint_dir is not None and (epoch + 1) % save_interval == 0:
            ckpt_path = Path(checkpoint_dir) / f"convaae_checkpoint_epoch{epoch + 1:03d}.pt"
            save_convautoencoder_checkpoint(model, optimizer, epoch, losses, ckpt_path)
            print(f"  Saved checkpoint: {ckpt_path.name}")
        
        # Save best model checkpoint
        if checkpoint_dir is not None and avg_loss == best_loss:
            best_path = Path(checkpoint_dir) / "convaae_best.pt"
            save_convautoencoder_checkpoint(model, optimizer, epoch, losses, best_path)
    
    print(f"Training complete. Final loss: {losses[-1]:.6f}, Best loss: {best_loss:.6f}")
    return model, losses
