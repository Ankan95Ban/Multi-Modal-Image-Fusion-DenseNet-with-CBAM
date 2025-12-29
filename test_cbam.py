import argparse
import os
import time
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.utils import save_image
import model_attention_dense

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--ir_dataroot", default="./datasets/test/IR", type=str)
parser.add_argument("--vis_dataroot", default="./datasets/test/VIS", type=str)
parser.add_argument("--output_root", default="./results/", type=str)
# UPDATED: Default set to your correct weight file
parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/fusion_last_10.pth")
parser.add_argument("--gpu_id", type=str, default="0")

if __name__ == "__main__":
    opt = parser.parse_args()
    
    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)

    # Load Model
    net = model_attention_dense.CBAMFuse().to(device)
    
    try:
        net.load_state_dict(torch.load(opt.checkpoint_path, map_location=device))
        print(f"Loaded checkpoint: {opt.checkpoint_path}")
    except FileNotFoundError:
        print(f"Error: Checkpoint not found at {opt.checkpoint_path}")
        exit()
        
    net.eval()
    
    transform = transforms.Compose([transforms.ToTensor()])
    
    # Get list of images (Assumes IR and VIS folders have matching filenames)
    if not os.path.exists(opt.ir_dataroot):
        print(f"Error: IR dataset path {opt.ir_dataroot} does not exist.")
        exit()
        
    image_files = [f for f in os.listdir(opt.ir_dataroot) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    print(f"Found {len(image_files)} images to process.")

    with torch.no_grad():
        for filename in image_files:
            start = time.time()
            
            # Construct paths
            ir_path = os.path.join(opt.ir_dataroot, filename)
            vis_path = os.path.join(opt.vis_dataroot, filename)
            
            if not os.path.exists(vis_path):
                print(f"Warning: Corresponding visible image for {filename} not found. Skipping.")
                continue

            # Load Images
            infrared = Image.open(ir_path).convert('L')
            visible = Image.open(vis_path).convert('L')
            
            infrared_t = transform(infrared).unsqueeze(0).to(device)
            visible_t = transform(visible).unsqueeze(0).to(device)
            
            # Inference
            fused_img = net(infrared_t, visible_t)
            
            # Add Residual (as per your original logic)
            fused_img = fused_img + infrared_t
            
            # Save Output
            save_path = os.path.join(opt.output_root, filename)
            save_image(fused_img.cpu(), save_path)
            
            end = time.time()
            print(f'Processed {filename} in {end-start:.4f}s')

    print("Testing Complete.")
