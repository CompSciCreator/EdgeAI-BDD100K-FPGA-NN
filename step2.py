import torch
from torch import nn

# Neural network for object detection
class ObjectDetectionNetwork(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Convolutional layers for feature extraction
    

        self.conv_stack = nn.Sequential(
            
			#3 input channels of different RGB colors
            #32 output channels capturing edges and textures
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            
			#activation function
            nn.ReLU(),
            
			#down sampling for more efficency
            nn.MaxPool2d(kernel_size=2, stride=2),

			#we add more channels allowing for more detailed learning
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            
			#activation function
            nn.ReLU(),
            
            # Downsampling
            nn.MaxPool2d(kernel_size=2, stride=2),

			#adding even more channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Detection head for classification
        self.cls_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            # Class prediction
            nn.Conv2d(64, num_classes, kernel_size=1) 
        )
        
        # Detection head for bounding box regression
        self.box_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),  # Reduce to 64 feature maps
            nn.ReLU(), # Activation function
            
			#The output has 4 channels representing the bounding box coordinates — (x, y, w, h).
            nn.Conv2d(64, 4, kernel_size=1) # Bounding box (x, y, w, h)
        )

    def forward(self, x):
        x = self.conv_stack(x) # Pass input through convolutional layers
        class_logits = self.cls_head(x)  # Get class predictions
        box_logits = self.box_head(x) # Get bounding box predictions
        return class_logits, box_logits # Return both outputs
    
# Define device and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize object detection model
model = ObjectDetectionNetwork(num_classes=10).to(device)
print(device)
print(model)

