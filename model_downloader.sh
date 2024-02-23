#!/bin/bash

# # Check if an output directory was provided
# if [ "$#" -ne 1 ]; then
#     echo "Usage: $0 <output_directory>"
#     exit 1
# fi

# Get the output directory from the script arguments
OUTPUT_DIR=.

# Create the output directory if it does not exist
if [ ! -d "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
fi

# Define model URLs and target filenames
MODEL_URL1="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C-8KFFT-InstVoc_HQ.ckpt"
TARGET_FILE_NAME1="MDX23C-8KFFT-InstVoc_HQ.ckpt"

MODEL_URL2="https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/model_vocals_segm_models_sdr_9.77.ckpt"
TARGET_FILE_NAME2="model_vocals_segm_models_sdr_9.77.ckpt"

MODEL_URL3="https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Voc_FT.onnx"
TARGET_FILE_NAME3="UVR-MDX-NET-Voc_FT.onnx"

# Download the first model
echo "Downloading $TARGET_FILE_NAME1 to $OUTPUT_DIR..."
wget $MODEL_URL1 -O "$OUTPUT_DIR/$TARGET_FILE_NAME1"
echo "Downloaded $TARGET_FILE_NAME1"

# Download the second model
echo "Downloading $TARGET_FILE_NAME2 to $OUTPUT_DIR..."
wget $MODEL_URL2 -O "$OUTPUT_DIR/$TARGET_FILE_NAME2"
echo "Downloaded $TARGET_FILE_NAME2"

# Download the third model
echo "Downloading $TARGET_FILE_NAME3 to $OUTPUT_DIR..."
wget $MODEL_URL3 -O "$OUTPUT_DIR/$TARGET_FILE_NAME3"
echo "Downloaded $TARGET_FILE_NAME3"

echo "All models downloaded successfully to $OUTPUT_DIR."
