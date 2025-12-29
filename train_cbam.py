import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from pytorch_ssim import ssim, tv_loss  # Ensure you have this file/module
from input_dataMed import ImageDataset   # Ensure you upload this next
import model_attention_dense             # The model file you uploaded previously

# Configuration
parser = argparse.ArgumentParser()
parser.add_argument("--ir_dataroot", default="./datasets/train/IR", type=str, help="Path to Infrared training images")
parser.add_argument("--vis_dataroot", default="./datasets/train/VIS", type=str, help="Path to Visible training images")
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--output_root", default="./training_outputs/", type=str)
parser.add_argument("--image_size", type=int, default=256)
parser.add_argument("--epoch", type=int, default=20)
parser.add_argument("--lr", type=float, default=0.0001)
parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints/")
parser.add_argument("--gpu_id", type=str, default="0")

if __name__ == "__main__":
    opt = parser.parse_args()
    
    # Set Device
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    # Create Directories
    if not os.path.exists(opt.checkpoint_dir):
        os.makedirs(opt.checkpoint_dir)
    if not os.path.exists(opt.output_root):
        os.makedirs(opt.output_root)

    # Initialize Model
    net = model_attention_dense.CBAMFuse().to(device)
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=opt.lr)
    
    # Data Loader
    # Note: Ensure ImageDataset in input_dataMed.py accepts these arguments
    train_datasets = ImageDataset(opt.ir_dataroot, opt.vis_dataroot, opt.image_size)
    dataloader = DataLoader(train_datasets, batch_size=opt.batch_size, shuffle=True)
    
    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total parameters: {total_params}')
    
    loss_weight = 10
    
    # Training Loop
    print("Start Training...")
    for epoch in range(opt.epoch):
        start = time.time()
        runloss = 0.0
        
        for index, data in enumerate(dataloader):
            # Assumes data is [ir, vis]
            ir = data[0].to(device)
            vis = data[1].to(device)
            
            # Forward Pass
            fused_img = net(ir, vis)
            
            # Loss Calculation (SSIM + TV)
            # Note: Ensure your ssim implementation supports 3 arguments if that was intended
            # Otherwise standard SSIM usually compares (output, target)
            LOSS_SSIM = 1 - ssim(fused_img, ir, vis) 
            LOSS_TV = tv_loss(fused_img - vis)
            loss = loss_weight * LOSS_SSIM + LOSS_TV
            
            runloss += loss.item()
            
            # Optimization
            optim.zero_grad()
            loss.backward()
            optim.step()

            if index % 10 == 0: # Print every 10 batches to reduce clutter
                print(f'Epoch [{epoch+1}/{opt.epoch}], Step [{index+1}/{len(dataloader)}], '
                      f'SSIM Loss: {LOSS_SSIM.item():.5f}, TV Loss: {LOSS_TV.item():.5f}, '
                      f'Total: {loss.item():.5f}')

        end = time.time()
        print(f'Epoch {epoch+1} time: {end-start:.2f}s')
        
        # Save Checkpoint
        torch.save(net.state_dict(), os.path.join(opt.checkpoint_dir, f'fusion_epoch_{epoch+1}.pth'))

    print("Training Complete.")
