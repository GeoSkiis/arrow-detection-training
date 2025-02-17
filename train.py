import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.transforms import functional as F
import torchvision.transforms as T
import numpy as np
import torchvision
import os
from PIL import Image, ImageDraw
import json

# Custom dataset class for arrows
class ArrowsDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted([f for f in os.listdir(os.path.join(root, "images")) if f.endswith(".png")]))
        self.anns = list(sorted([f for f in os.listdir(os.path.join(root, "annotations")) if f.endswith(".json")]))

        # Debugging: print number of images and annotations
        print(f"Number of images found: {len(self.imgs)}")
        print(f"Number of annotations found: {len(self.anns)}")

    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")

        # Load corresponding annotation
        ann_path = os.path.join(self.root, "annotations", self.anns[idx])
        with open(ann_path) as f:
            annotation = json.load(f)

        # Create masks from annotation polygons
        masks = []
        boxes = []
        for shape in annotation['shapes']:
            if shape['label'] == "arrow":
                mask = Image.new('L', (img.width, img.height))
                mask_draw = ImageDraw.Draw(mask)
                
                # Ensure points are correctly formatted as tuples of integers/floats
                polygon = [(int(point[0]), int(point[1])) for point in shape['points']]
                
                # Draw the polygon on the mask
                mask_draw.polygon(polygon, outline=1, fill=1)
                masks.append(np.array(mask))
                
                # Create bounding box from polygon points
                points = np.array(shape['points'])
                xmin = np.min(points[:, 0])
                xmax = np.max(points[:, 0])
                ymin = np.min(points[:, 1])
                ymax = np.max(points[:, 1])
                boxes.append([xmin, ymin, xmax, ymax])

        # Convert the list of numpy arrays to a single numpy array and then to a torch tensor
        masks = np.array(masks)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(boxes),), dtype=torch.int64)

        # Target dictionary
        target = {"boxes": boxes, "labels": labels, "masks": masks}

        if self.transforms:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.imgs)


# Transformation function
def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def get_instance_segmentation_model(num_classes):
    # Load the pre-trained Mask R-CNN model with the latest weights
    weights = MaskRCNN_ResNet50_FPN_Weights.COCO_V1
    model = maskrcnn_resnet50_fpn(weights=weights)

    # Get the number of input features for the box classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the box_predictor (for the detection head) with a new one for your number of classes
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)

    # Now that we have changed the box head, we will also replace the mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256

    # Mask predictor head
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask,
        hidden_layer,
        num_classes
    )

    return model



# Custom collate function to handle variable-sized targets
def collate_fn(batch):
    return tuple(zip(*batch))


# Train the model for one epoch
def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    model.train()
    for i, (images, targets) in enumerate(data_loader):
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # Backward pass and optimizer step
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if i % print_freq == 0:
            print(f"Epoch [{epoch}], Iteration [{i}]: Loss = {losses.item()}")


# Evaluate the model (on the test set)
def evaluate(model, data_loader, device):
    model.eval()
    print("Evaluating...")
    with torch.no_grad():
        for images, targets in data_loader:
            images = list(img.to(device) for img in images)
            predictions = model(images)
            # Here you can print or log predictions for evaluation

if __name__ == "__main__":
    # Set number of classes (1 for background, 1 for the arrows)
    num_classes = 2  # 1 background + 1 for the arrows

    # Initialize the model
    model = get_instance_segmentation_model(num_classes)

    # Use GPU if available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Load the dataset
    dataset = ArrowsDataset(root="dataset", transforms=get_transform(train=True))
    dataset_test = ArrowsDataset(root="dataset", transforms=get_transform(train=False))

    # Split dataset into training and testing (adjusted for 10 images)
    indices = torch.randperm(len(dataset)).tolist()
    dataset_train = torch.utils.data.Subset(dataset, indices[:80])  # First 8 images for training
    dataset_test = torch.utils.data.Subset(dataset, indices[80:])   # Last 2 images for testing

    # Create data loaders (set num_workers=0 on Windows to avoid multiprocessing issues)
    data_loader = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=0, collate_fn=collate_fn)
    data_loader_test = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # Optimizer and learning rate scheduler
    params = [p for p in model.parameters() if p.requires_grad]
    # optimizer = optim.SGD(params, lr=0.001, momentum=0.9, weight_decay=0.001)
    optimizer = optim.Adam(params, lr=0.0005, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.1)

    # Number of epochs
    num_epochs = 10

    # Train the model and save after each epoch
    for epoch in range(num_epochs):
        # Train for one epoch
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the test dataset
        evaluate(model, data_loader_test, device=device)

        # Save the model's state_dict and optimizer's state_dict
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, f'mask_rcnn_arrows_epoch_{epoch+1}.pth')

    print("Training complete.")