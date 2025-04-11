import torch
import numpy
import pandas
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import json
import os
from PIL import Image

# Define the BDD100KDetectionDataset class
class BDD100KDetectionDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):

        # stores the image path
        self.image_dir = image_dir
        self.transform = transform
        self.data = []

        # Load label data
        # path to the json file
        with open(label_path, 'r') as f:

            #parse through the json data
            self.labels = json.load(f)

        # Extract necessary information
        # iterates over the json entries
        for item in self.labels:
            #file name for the item in json
            image_path = os.path.join(image_dir, item['name'])

            #gets the extracted item label
            annotations = item.get('labels', [])

            #stores the image paths
            self.data.append((image_path, annotations))


     # this defines our sample imgs in BDD100KDetectionDataset
    def __len__(self):
        return len(self.data)

    # using the index we extract specific items
    def __getitem__(self, idx):
        image_path, annotations = self.data[idx]

        # Loads our image and makes sure its in RGB format
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            # If a transform (e.g., resizing, normalization, data augmentation)
            # is provided using it will be apllied to our image
            image = self.transform(image)

        # Process annotations (bounding boxes and class labels)
        boxes = []
        labels = []
        for ann in annotations:

            # loop through each (ann) in the annotation list
            bbox = ann.get('box2d', {})
            if bbox:
                # extract bound box information
                boxes.append([bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']])
                # either a category is determined or unknown is returned
                labels.append(ann.get('category', 'unknown'))

        #returning the processed image as a tensor along with box coordinates and a label
        return image, {"boxes": torch.tensor(boxes, dtype=torch.float32), 
                        "labels": labels}

# Custom collate function for variable-sized data
def collate_fn(batch):
    images = []
    targets = []

    #combine our image and labels into a single batch
    for img, target in batch:
        images.append(img)
        targets.append(target)
    return images, targets

# Paths (Modify based on dataset location) Mikes location
IMAGE_DIR = "/Users/andrewpaolella/Desktop/Final-Project/BDD100k/bdd100k/bdd100k/images/100k/train"
LABEL_PATH = "/Users/andrewpaolella/Desktop/Final-Project/BDD100k/labels/det_v2_train_release.json"

# amount of pictures going through in one batch
batch_size = 8

# defining transformations to convert PIL img's (an image in memory)
# to Tensors
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # makes images smaller & faster
    transforms.ToTensor()
])

# Create training and test datasets with transformations
training_data = BDD100KDetectionDataset(IMAGE_DIR, LABEL_PATH, transform=transform)
test_data = BDD100KDetectionDataset(IMAGE_DIR, LABEL_PATH, transform=transform)
# Limit dataset size for faster testing
training_data.data = training_data.data[:512]
test_data.data = test_data.data[:128]

# Create DataLoaders with the custom collate function
train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


# had to change this because x & y are lists and we need tensors instead
#

#------------------------------------#
# for X, y in test_dataloader:
#     print(f"Shape of X [N,C,H,W]: {X.shape}")
#     print(f"Shape of y: {y.shape} {y.dtype}")
#     break
#------------------------------------#

# Print example data
for X, y in test_dataloader:
    print(f"Number of images: {len(X)}\n")
    print(f"First Image Shape: {X[0].shape}\n")
    print(f"First Target: {y[0]}\n")
    break
