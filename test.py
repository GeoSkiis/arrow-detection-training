import torch
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torchvision.models.detection import maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
import torchvision
import numpy as np
import cv2

# Load the trained model
def load_model(model_path, num_classes, device):
    # Load a pre-trained Mask R-CNN model with no weights
    model = maskrcnn_resnet50_fpn(weights=None)
    
    # Get the number of input features for the box classifier
    in_features_box = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the box predictor head
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features_box, num_classes)
    
    # Replace the mask predictor head
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = torchvision.models.detection.mask_rcnn.MaskRCNNPredictor(
        in_features_mask, hidden_layer, num_classes
    )

    # Load the checkpoint and extract only the model's state_dict
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])  # Load the model state dict
    
    # Move the model to the appropriate device (GPU or CPU)
    model.to(device)
    
    # Set the model to evaluation mode
    model.eval()
    
    return model

# Function to remove background using model prediction
def remove_background(model, image_path, output_path, device):
    # Load the image
    img = Image.open(image_path).convert("RGB")
    
    # Apply the necessary transformations
    transform = T.Compose([T.ToTensor()])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension
    
    # Move the image tensor to the appropriate device (GPU or CPU)
    img_tensor = img_tensor.to(device)
    
    # Run the model on the input image
    with torch.no_grad():
        prediction = model(img_tensor)[0]
    
    # Get the masks from the prediction
    # masks = prediction['masks'] > 0.5  # Threshold for masks
    # masks = masks.squeeze(1).cpu().numpy()  # Convert masks to NumPy arrays on CPU
    # Adjust thresholding
    masks = prediction['masks'] > 0.5

    # Filter by confidence score
    confidence_threshold = 0.4
    valid_indices = [i for i, score in enumerate(prediction['scores'].cpu().numpy()) if score > confidence_threshold]
    masks = masks[valid_indices].squeeze(1).cpu().numpy()
    
    # Combine all masks into a single mask
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        combined_mask = np.maximum(combined_mask, mask.astype(np.uint8))
    # kernel = np.ones((5, 5), np.uint8)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel)
    # combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel)
    
    # Create an image where the background is removed (transparent)
    img_array = np.array(img)
    img_array[combined_mask == 0] = 0  # Set background to black (you can change it to transparent or other)
    
    # Save the image with the background removed
    output_img = Image.fromarray(img_array)
    output_img.save(output_path)
    print(f"Saved image with background removed to {output_path}")

# Main function to run the test
if __name__ == "__main__":
    # Path to the trained model
    model_path = "mask_rcnn_arrows_epoch_10.pth"
    
    # Path to the test image and the output image
    test_image_path = "test.png"
    output_image_path = "test_no_background7_1.png"
    
    # Number of classes (background + arrow class)
    num_classes = 2  # 1 background + 1 arrow
    
    # Set the device to GPU if available, else fallback to CPU
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    # Load the trained model
    model = load_model(model_path, num_classes, device)
    test_image = cv2.imread(test_image_path)
    # Perform background removal
    remove_background(model, test_image, output_image_path, device)