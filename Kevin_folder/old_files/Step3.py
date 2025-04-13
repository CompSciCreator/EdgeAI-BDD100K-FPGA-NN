# Tite: Step3.py
# Description: This script trains the ObjectDetectionNetwork for the BDD100K dataset. 
# It loads the dataset from Step1.py using train_dataloaderand imports the model from step2.py (ObjectDetectionNetwork).
# Sets up sample training, defines a loss function (CrossEntropyLoss) and optimizer.
# needs work or scrap altogether? 
# Author: Andrew Paolella

import torch
from torch import nn, optim
from step1 import train_dataloader, test_dataloader
from step2 import ObjectDetectionNetwork
from datetime import datetime

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Label mapping
category_map = {
    'car': 0, 'bus': 1, 'truck': 2, 'person': 3, 'rider': 4,
    'bike': 5, 'motor': 6, 'traffic light': 7, 'traffic sign': 8, 'other': 9
}

# Model setup
model = ObjectDetectionNetwork(num_classes=10).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Logging
log_file = open("step3_10epoch.txt", "w")
start_time = datetime.now()

def log(msg):
    print(msg)
    log_file.write(msg + "\n")

def train_eval(dataloader, model, criterion, optimizer=None):
    model.train() if optimizer else model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.set_grad_enabled(optimizer is not None):
        for images, targets in dataloader:
            try:
                images = torch.stack(images).to(device)
            except:
                continue

            labels = [category_map.get(t['labels'][0], 9) for t in targets]
            labels = torch.tensor(labels, dtype=torch.long).to(device)

            if optimizer:
                optimizer.zero_grad()

            class_logits, _ = model(images)
            preds = class_logits.mean([2, 3])
            loss = criterion(preds, labels)

            if optimizer:
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            correct += (preds.argmax(1) == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total if total > 0 else 0
    return accuracy, avg_loss

# Training loop
try:
    epochs = 10
    for epoch in range(1, epochs + 1):
        train_acc, train_loss = train_eval(train_dataloader, model, criterion, optimizer)
        log(f"Epoch {epoch}")
        log(f"Accuracy: {train_acc:.4f}")
        log(f"Avg loss: {train_loss:.6f}\n")

except KeyboardInterrupt:
    log("\nTraining manually stopped.")

end_time = datetime.now()
log(f"Total time: {end_time - start_time}")
log_file.close()
