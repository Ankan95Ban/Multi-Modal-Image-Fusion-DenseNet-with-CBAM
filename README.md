# Improved Multi-modal Image Fusion with Attention and Dense Networks

This repository contains the official PyTorch implementation of the paper:
**"Improved Multi-modal Image Fusion with Attention and Dense Networks: Visual and Quantitative Evaluation"**
*Published in: Communications in Computer and Information Science, Springer (2024)*

[**Link to Paper**](https://link.springer.com/chapter/10.1007/978-3-031-58535-7_20)

## 📝 Abstract
This paper proposes a novel deep learning architecture for fusing multi-modal images (such as Infrared and Visible). The model integrates **DenseNet** blocks for robust feature extraction and **Convolutional Block Attention Modules (CBAM)** to focus on salient spatial and channel-wise features. The approach demonstrates superior performance in both visual quality and quantitative metrics compared to existing state-of-the-art methods.

## 🏗️ Architecture
The network consists of:
1.  **Dual-Branch Feature Extraction:** Two DenseNet branches process Infrared and Visible images independently.
2.  **Attention Mechanism:** CBAM blocks refine the features by emphasizing important channels and spatial regions.
3.  **Reconstruction:** A series of convolutional layers merge the features to generate the final fused image.

## 📂 Project Structure

. ├── checkpoints/ # Saved model weights ├── datasets/ # Dataset directory │ ├── train/ │ │ ├── IR/ │ │ └── VIS/ │ └── test/ │ ├── IR/ │ └── VIS/ ├── model_attention_dense.py # Model architecture (CBAMFuse) ├── input_data.py # Dataloader ├── pytorch_ssim.py # SSIM Loss function ├── train_cbam.py # Training script ├── test_cbam.py # Testing/Inference script └── requirements.txt # Dependencies


## 🚀 Getting Started

### Prerequisites
Install the required dependencies:
```bash
pip install -r requirements.txt

Dataset Preparation
Organize your data into train and test folders. Ensure that Infrared (IR) and Visible (VIS) images have matching filenames or are sorted alphabetically.

Training
To train the model from scratch:
python train_cbam.py --ir_dataroot ./datasets/train/IR --vis_dataroot ./datasets/train/VIS --epoch 20

Testing / Inference
To test the model using pre-trained weights:
python test_cbam.py --ir_dataroot ./datasets/test/IR --vis_dataroot ./datasets/test/VIS --output_root ./results/

If you find this work useful in your research, please cite:
@InProceedings{10.1007/978-3-031-58535-7_20,
author="Banerjee, Ankan
and Patra, Dipti
and Roy, Pradipta",
title="Improved Multi-modal Image Fusion with Attention and Dense Networks: Visual and Quantitative Evaluation",
booktitle="Communications in Computer and Information Science",
year="2024",
publisher="Springer Nature Switzerland",
pages="242--255",
isbn="978-3-031-58535-7"
}




