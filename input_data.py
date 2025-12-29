import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, ir_dataroot, vis_dataroot, image_size=256):
        """
        Args:
            ir_dataroot (str): Path to Infrared images.
            vis_dataroot (str): Path to Visible images.
            image_size (int): Size to resize/crop images (if needed in transforms).
        """
        self.ir_dataroot = ir_dataroot
        self.vis_dataroot = vis_dataroot
        self.image_size = image_size
        
        # Get list of images and sort them to ensure matching pairs
        self.ir_images = sorted([
            f for f in os.listdir(ir_dataroot) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        self.vis_images = sorted([
            f for f in os.listdir(vis_dataroot) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
        ])
        
        # Validation
        if len(self.ir_images) != len(self.vis_images):
            print(f"Warning: Number of IR images ({len(self.ir_images)}) "
                  f"does not match Visible images ({len(self.vis_images)}).")
        
        # Use the smaller length to avoid errors
        self.dataset_len = min(len(self.ir_images), len(self.vis_images))
        
        # Standard transform
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, index):
        # Construct paths
        ir_path = os.path.join(self.ir_dataroot, self.ir_images[index])
        vis_path = os.path.join(self.vis_dataroot, self.vis_images[index])
        
        # Open images (Convert to Grayscale 'L')
        img_ir = Image.open(ir_path).convert('L')
        img_vis = Image.open(vis_path).convert('L')
        
        # Apply transforms
        img_ir = self.transform(img_ir)
        img_vis = self.transform(img_vis)
        
        return img_ir, img_vis

# Test block
if __name__ == "__main__":
    # Example usage
    # Ensure these paths exist before running this block directly
    ir_path = "./datasets/train/IR"
    vis_path = "./datasets/train/VIS"
    
    if os.path.exists(ir_path) and os.path.exists(vis_path):
        dataset = ImageDataset(ir_path, vis_path)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
        
        print(f"Dataset length: {len(dataset)}")
        for i, (ir, vis) in enumerate(dataloader):
            print(f"Batch {i}: IR Shape {ir.shape}, VIS Shape {vis.shape}")
            if i >= 2: break 
    else:
        print("Set valid paths in __main__ to test the dataloader.")
