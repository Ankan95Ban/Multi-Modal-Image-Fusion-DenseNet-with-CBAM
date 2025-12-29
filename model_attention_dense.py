import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt

"""
Paper: Improved Multi-modal Image Fusion with Attention and Dense Networks
Authors: Ankan Banerjee, Dipti Patra, Pradipta Roy
Published in: Communications in Computer and Information Science, Springer (2024)
"""

class ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.se = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result = self.maxpool(x)
        avg_result = self.avgpool(x)
        max_out = self.se(max_result)
        avg_out = self.se(avg_result)
        output = self.sigmoid(max_out + avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_result, _ = torch.max(x, dim=1, keepdim=True)
        avg_result = torch.mean(x, dim=1, keepdim=True)
        result = torch.cat([max_result, avg_result], 1)
        output = self.conv(result)
        # print(output.shape, 'spatial_debug') 
        output = self.sigmoid(output)
        return output

class CBAMBlock(nn.Module):
    def __init__(self, channel=512, reduction=16, kernel_size=3):
        super().__init__()
        self.ca = ChannelAttention(channel=channel, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=kernel_size)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x
        out = x * self.ca(x)
        mul = self.sa(out)
        out = out * mul
        return out + residual

class CBAMFuse(nn.Module):
    def __init__(self):
        super(CBAMFuse, self).__init__()
        # Dense Block 1 (Infrared Branch)
        self.conv1_1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU())
        self.conv1_2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())
        self.conv1_3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU())
        self.conv1_4 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1), nn.ReLU())
        
        # Dense Block 2 (Visible Branch)
        self.conv2_1 = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1), nn.ReLU())
        self.conv2_2 = nn.Sequential(nn.Conv2d(16, 16, 3, 1, 1), nn.ReLU())
        self.conv2_3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU())
        self.conv2_4 = nn.Sequential(nn.Conv2d(48, 16, 3, 1, 1), nn.ReLU())
        
        # Reconstruction Block
        self.conv3_1 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.ReLU())
        self.conv3_2 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.ReLU())
        self.conv3_3 = nn.Sequential(nn.Conv2d(32, 16, 3, 1, 1), nn.ReLU())
        self.conv3_4 = nn.Sequential(nn.Conv2d(16, 1, 3, 1, 1), nn.ReLU())
        
        self.CBAMBlock = CBAMBlock(channel=64, reduction=16, kernel_size=3)

    def forward(self, infrared, visible):
        # Infrared Branch
        c1_1 = self.conv1_1(infrared)
        c1_2 = self.conv1_2(c1_1)
        c1_3 = self.conv1_3(torch.cat((c1_2, c1_1), dim=1))
        c1_4 = self.conv1_4(torch.cat((c1_3, c1_2, c1_1), dim=1))
        fuse_ir = torch.cat((c1_4, c1_3, c1_2, c1_1), dim=1)
        
        infraAtten = self.CBAMBlock(fuse_ir)
        
        # Visible Branch
        c2_1 = self.conv2_1(visible)
        c2_2 = self.conv2_2(c2_1)
        c2_3 = self.conv2_3(torch.cat((c2_2, c2_1), dim=1))
        c2_4 = self.conv2_4(torch.cat((c2_3, c2_2, c2_1), dim=1))
        fuse_vis = torch.cat((c2_4, c2_3, c2_2, c2_1), dim=1)
        
        visibleAtten = self.CBAMBlock(fuse_vis)
        
        # Fusion
        concate = torch.cat((visibleAtten, infraAtten), dim=1)
        
        # Reconstruction
        c3_1 = self.conv3_1(concate)
        c3_2 = self.conv3_2(c3_1)
        c3_3 = self.conv3_3(c3_2)
        c3_4 = self.conv3_4(c3_3)
        return c3_4

# Example Usage
if __name__ == "__main__":
    # Check for CUDA
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Running on: {device}")

    # Initialize Model
    net = CBAMFuse().to(device)
    
    # Load dummy images or real images if paths exist
    try:
        # Update these paths to valid images on your system or use placeholder logic
        ir_path = 'datasets/IR/ir75.png' 
        vis_path = 'datasets/VIS/vis75.png'
        
        inputir = Image.open(ir_path).convert('L') # Ensure grayscale
        inputvis = Image.open(vis_path).convert('L')
        
        t1 = transforms.ToTensor()
        t2 = transforms.ToPILImage()
        
        input_tensor1 = t1(inputir).unsqueeze(0).to(device)
        input_tensor2 = t1(inputvis).unsqueeze(0).to(device)
        
        with torch.no_grad():
            out = net(input_tensor1, input_tensor2)
            outfinal = out.squeeze(0).cpu()
            
        print("Output shape:", outfinal.shape)
        out_image = t2(outfinal)
        plt.imshow(out_image, cmap='gray')
        plt.show()
        
    except FileNotFoundError:
        print("Sample images not found. Please ensure 'datasets/IR/ir75.png' exists to test.")
