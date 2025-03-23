#Filename: Step1.py
#Description: Trying to build and create our dataset
#Programmed by: Andrew Paolella, (add here)

import torch
import numpy 
import pandas 
from torch.utils.data import DataLoade  r, Dataset
from torchvision import transforms
import json
import os
from PIL import Image
import torch.utils.data


# Paths (Modify based on dataset location)
IMAGE_DIR = "./Users/andrewpaolella/vsprojects/Edge-AI/Final-Project/archive/bdd100k/bdd100k/images/100k/train"
LABEL_PATH = "./Users/andrewpaolella/vsprojects/Edge-AI/Final-Project/archive/labels/det_v2_train_release.json"

# Dataset & DataLoader
batch_size = 64

# Download training data from open datasets
training_data = BDD100KDetectionDataset(IMAGE_DIR, LABEL_PATH)

# Download test data from open datasets
test_data = BDD100KDetectionDataset(IMAGE_DIR, LABEL_PATH)

train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

for X, y in test_dataloader:
    print(f"Shape of X [N,C,H,W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
