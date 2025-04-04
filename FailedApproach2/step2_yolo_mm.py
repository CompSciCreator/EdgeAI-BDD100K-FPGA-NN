import torch
from torch import nn
from torchvision.models import resnet50, ResNet50_Weights


class YOLOv8ResNet50(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(YOLOv8ResNet50, self).__init__()
        self.num_classes = num_classes
        
        # Load pretrained ResNet50
        backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        
        # Feature extraction layers
        self.conv1 = backbone.conv1
        self.bn1 = backbone.bn1
        self.relu = backbone.relu
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # Output: 256 channels
        self.layer2 = backbone.layer2  # Output: 512 channels
        self.layer3 = backbone.layer3  # Output: 1024 channels
        self.layer4 = backbone.layer4  # Output: 2048 channels
        
        # Detection heads
        self.head_small = self._make_head(512, num_classes)   # For small objects (from layer2)
        self.head_medium = self._make_head(1024, num_classes) # For medium objects (from layer3)
        self.head_large = self._make_head(2048, num_classes)  # For large objects (from layer4)
        
    def _make_head(self, in_channels, num_classes):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels//2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels//2),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels//2, 3*(5 + num_classes), kernel_size=1)
        )
    
    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        # Initial layers
        x = self.conv1(x)       # [B, 64, H/2, W/2]
        # print(f"After conv1: {x.shape}")
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # [B, 64, H/4, W/4]
        
        # Backbone features
        c1 = self.layer1(x)     # [B, 256, H/4, W/4]
        c2 = self.layer2(c1)    # [B, 512, H/8, W/8]  - for small objects
        c3 = self.layer3(c2)    # [B, 1024, H/16, W/16] - for medium objects
        c4 = self.layer4(c3)    # [B, 2048, H/32, W/32] - for large objects
        
        # Detection heads
        small_out = self.head_small(c2)   # For small objects
        medium_out = self.head_medium(c3) # For medium objects
        large_out = self.head_large(c4)   # For large objects
        
        return [small_out, medium_out, large_out]

# Define device and initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = YOLOv8ResNet50(num_classes=10).to(device)
print(device)
print(model)