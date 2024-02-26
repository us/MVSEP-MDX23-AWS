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

!wget "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/model_2_stem_full_band_8k.yaml"

!wget "https://github.com/ZFTurbo/Music-Source-Separation-Training/releases/download/v1.0.0/config_vocals_segm_models.yaml"

echo "All models downloaded successfully to $OUTPUT_DIR."

# #!/bin/bash
# mkdir demucs_repo
# cd demucs_repo
# # MDX Models
# mdx_models=(
#     "0d19c1c6-0f06f20e.th"
#     "5d2d6c55-db83574e.th"
#     "7d865c68-3d5dd56b.th"
#     "7ecf8ec1-70f50cc9.th"
#     "a1d90b5c-ae9d2452.th"
#     "c511e2ab-fe698775.th"
#     "cfa93e08-61801ae1.th"
#     "e51eebcc-c1b80bdd.th"
#     "6b9c2ca1-3fd82607.th"
#     "b72baf4e-8778635e.th"
#     "42e558d4-196e0e1b.th"
#     "305bc58f-18378783.th"
#     "14fc6a69-a89dd0ee.th"
#     "464b36d7-e5a9386e.th"
#     "7fd6ef75-a905dd85.th"
#     "83fc094f-4a16d450.th"
#     "1ef250f1-592467ce.th"
#     "902315c2-b39ce9c9.th"
#     "9a6b4851-03af0aa6.th"
#     "fa0cb7f9-100d8bf4.th"
# )

# # Hybrid Transformer models
# hybrid_transformer_models=(
#     "955717e8-8726e21a.th"
#     "f7e0c4bc-ba3fe64a.th"
#     "d12395a8-e57c48e6.th"
#     "92cfc3b6-ef3bcb9c.th"
#     "04573f0d-f3cf25b2.th"
#     "75fc33f5-1941ce65.th"
# )

# # Experimental 6 sources model
# experimental_model="5c90dfd2-34c22ccb.th"

# # Download function
# download_file() {
#     local url=$1
#     local filename=$(basename $url)
#     local modified_filename="${filename%%-*}.th"  # Extract the first part of the name and append ".th"
#     echo "Downloading $url..."
#     wget -q $url -O  "$OUTPUT_DIR/$modified_filename"
# }

# # Download MDX Models
# # mkdir -p ..
# # cd ..
# for model in "${mdx_models[@]}"; do
#     download_file "https://dl.fbaipublicfiles.com/demucs/mdx_final/$model"
# done
# # cd ..

# # Download Hybrid Transformer models
# # mkdir -p hybrid_transformer
# # cd hybrid_transformer
# for model in "${hybrid_transformer_models[@]}"; do
#     download_file "https://dl.fbaipublicfiles.com/demucs/hybrid_transformer/$model"
# done
# # cd ..

# # Download Experimental model
# download_file "https://dl.fbaipublicfiles.com/demucs/experimental_models/$experimental_model"

# wget "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_ft.yaml"
# wget "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs.yaml"
# wget "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/htdemucs_6s.yaml"
# wget "https://raw.githubusercontent.com/facebookresearch/demucs/main/demucs/remote/hdemucs_mmi.yaml"

