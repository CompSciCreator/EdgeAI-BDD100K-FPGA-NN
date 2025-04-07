import torch
from torch import nn

class ObjectDetectionNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Convolutional layers for feature extraction
        self.conv_stack = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Detection head for classification
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

        # Detection head for bounding box regression
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 4, kernel_size=1)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        class_logits = self.cls_head(x)
        box_logits = self.box_head(x)
        return class_logits, box_logits

# Define device and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ObjectDetectionNetwork(num_classes=10).to(device)
print(device)
print(model)