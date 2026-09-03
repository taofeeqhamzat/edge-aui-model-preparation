"""
training.py
Foundational PyTorch training loop for the GRU model (Edge-AUI Framework).
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from preprocessing import MicroInteractionSequenceDataset, FEATURE_NAMES

# Configuration
DATA_DIR = "../.data/raw"
BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 1e-3
HIDDEN_DIM = 32
NUM_LAYERS = 2
NUM_CLASSES = 6  # IDLE, CLICK, FORM_SUBMIT, BACKTRACK, RAPID_SCROLL, HOVER_DWELL

class EdgeAUIGRU(nn.Module):
    """
    Lightweight GRU for edge inference. 
    The base layers encode kinematics (foundational priors) and structural patterns.
    """
    def __init__(self, input_dim, hidden_dim, num_layers, num_classes):
        super(EdgeAUIGRU, self).__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def train_foundation_model():
    print(f"Loading datasets from {DATA_DIR}...")
    dataset = MicroInteractionSequenceDataset(data_root=DATA_DIR)
    
    if len(dataset) == 0:
        print(f"Warning: No valid sequences found in {DATA_DIR}. Ensure data is populated.")
        return
        
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = EdgeAUIGRU(
        input_dim=len(FEATURE_NAMES), 
        hidden_dim=HIDDEN_DIM, 
        num_layers=NUM_LAYERS, 
        num_classes=NUM_CLASSES
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    print(f"Starting Foundational Training for {EPOCHS} epochs over {len(dataset)} sequences...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * batch_x.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
            
        epoch_loss = total_loss / total
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
        
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/foundational_gru.pth"
    torch.save(model.state_dict(), model_path)
    print(f"Foundational model saved to {model_path}")

if __name__ == "__main__":
    train_foundation_model()
