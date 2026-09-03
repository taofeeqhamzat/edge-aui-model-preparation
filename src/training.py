"""
training.py
Foundational PyTorch training loop for the GRU model (Edge-AUI Framework).
"""

import os
import sys
import argparse
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

try:
    from preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES
except ImportError:
    from src.preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES

try:
    from data_manager import find_project_root, is_colab, is_kaggle
except ImportError:
    try:
        from src.data_manager import find_project_root, is_colab, is_kaggle
    except ImportError:
        find_project_root = lambda: os.getcwd()
        is_colab = lambda: False
        is_kaggle = lambda: False

# Default Configuration
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".data", "raw"))
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_CLASSES = 6  # IDLE, CLICK, FORM_SUBMIT, BACKTRACK, RAPID_SCROLL, HOVER_DWELL


def get_device(device_override: Optional[str] = None) -> torch.device:
    """
    Select optimal execution provider (CUDA > MPS > CPU).
    In hosted environments like Colab/Kaggle, CUDA is preferred.
    """
    if device_override:
        return torch.device(device_override)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EdgeAUIGRU(nn.Module):
    """
    Lightweight GRU for edge inference. 
    The base layers encode kinematics (foundational priors) and structural patterns.
    """
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_classes: int):
        super(EdgeAUIGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)


def train_foundation_model(
    data_dir: Optional[str] = None,
    hf_repo_id: str = "T40/edge-aui-framework-data",
    hf_token: Optional[str] = None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    hidden_dim: int = HIDDEN_DIM,
    num_layers: int = NUM_LAYERS,
    max_sequences: Optional[int] = None,
    max_files_per_dataset: Optional[int] = None,
    device: Optional[str] = None,
    output_dir: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Foundational training loop for EdgeAUIGRU.
    Supports local datasets as well as automatic hosted synchronization from Hugging Face Hub.
    """
    target_data_dir = data_dir if data_dir is not None else DATA_DIR
    target_device = get_device(device)

    if verbose:
        print(f"[Training] Target execution device: {target_device}")
        print(f"[Training] Loading interaction sequences (data_root='{target_data_dir}', repo='{hf_repo_id}')...")

    dataset = MicroInteractionSequenceDataset(
        data_root=target_data_dir,
        hf_repo_id=hf_repo_id,
        hf_token=hf_token,
        max_sequences=max_sequences,
        max_files_per_dataset=max_files_per_dataset
    )
    
    if len(dataset) == 0:
        print(f"[Training] Warning: No valid sequences found in {target_data_dir}. Ensure data is populated.")
        return {"model": None, "history": {"loss": [], "accuracy": []}}
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = EdgeAUIGRU(
        input_dim=len(FEATURE_NAMES), 
        hidden_dim=hidden_dim, 
        num_layers=num_layers, 
        num_classes=NUM_CLASSES
    ).to(target_device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

    if verbose:
        print(f"[Training] Starting Foundational Training: {epochs} epochs over {len(dataset)} sequences on {target_device}...")

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(target_device)
            batch_y = batch_y.to(target_device)

            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = total_loss / max(total, 1)
        epoch_acc = 100.0 * correct / max(total, 1)
        history["loss"].append(epoch_loss)
        history["accuracy"].append(epoch_acc)

        if verbose:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    # Resolve model output path
    if output_dir is None:
        proj_root = find_project_root()
        models_path = os.path.join(proj_root, "models")
    else:
        models_path = os.path.abspath(output_dir)

    os.makedirs(models_path, exist_ok=True)
    model_path = os.path.join(models_path, "foundational_gru.pth")
    torch.save(model.state_dict(), model_path)

    if verbose:
        print(f"[Training] Foundational model successfully saved to: {model_path}")

    return {
        "model": model,
        "model_path": model_path,
        "history": history,
        "dataset_size": len(dataset),
        "device": str(target_device)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Foundational GRU Training for Edge-AUI Framework")
    parser.add_argument("--data-dir", type=str, default=None, help="Path to raw dataset directory")
    parser.add_argument("--hf-repo", type=str, default="T40/edge-aui-framework-data", help="Hugging Face repo ID")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--max-sequences", type=int, default=None, help="Cap max sequences for fast iteration")
    parser.add_argument("--device", type=str, default=None, help="Execution device (cuda/mps/cpu)")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save checkpoints")
    args = parser.parse_args()

    train_foundation_model(
        data_dir=args.data_dir,
        hf_repo_id=args.hf_repo,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        max_sequences=args.max_sequences,
        device=args.device,
        output_dir=args.output_dir
    )
