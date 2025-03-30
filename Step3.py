# Tite: Step3.py
# Description: This script trains the ObjectDetectionNetwork for the BDD100K dataset. 
# It loads the dataset from Step1.py using train_dataloaderand imports the model from step2.py (ObjectDetectionNetwork).
# Sets up sample training, defines a loss function (CrossEntropyLoss) and optimizer.
# needs work or scrap altogether? 
# Author: Andrew Paolella

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

# Import dataset and model
from Step1 import train_dataloader, test_dataloader
from step2 import ObjectDetectionNetwork

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize model
model = ObjectDetectionNetwork(num_classes=10).to(device)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Optimizer(model.parameters(), lr=0.001)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for images, targets in train_dataloader:
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad()
        class_logits, box_logits = model(images)
        loss = criterion(class_logits, targets)  # Simplified loss for classification
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss:.4f}")


