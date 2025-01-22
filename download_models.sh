#!/bin/bash

# Create models directory
mkdir -p models
cd models

# Define model URLs and target filenames
echo "Downloading models..."

# Download MDX23C model
echo "Downloading MDX23C-8KFFT-InstVoc_HQ.ckpt..."
wget -q --show-progress https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt

# Download Vocals Segmentation model
echo "Downloading model_vocals_segm_models_sdr_9.77.ckpt..."
wget -q --show-progress https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt

# Download UVR-MDX-NET model
echo "Downloading UVR-MDX-NET-Voc_FT.onnx..."
wget -q --show-progress https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx

# Download config files
echo "Downloading configuration files..."
wget -q --show-progress https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml
wget -q --show-progress https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml

echo "All models and configs downloaded successfully!"
cd ..

# Make the script executable
chmod +x download_models.sh 