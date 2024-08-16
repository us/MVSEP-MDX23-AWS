# Use the SageMaker PyTorch image for INFERENCE
FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/pytorch-inference:2.1.0-gpu-py310

# Set environment variables
ENV PATH="/opt/ml/code:${PATH}"
ENV SAGEMAKER_SUBMIT_DIRECTORY /opt/ml/code
ENV MODEL_SERVER_TIMEOUT_ENV=600
ENV MODEL_SERVER_RESPONSE_TIMEOUT=600
ENV MODEL_SERVER_TIMEOUT=600
ENV SAGEMAKER_MODEL_SERVER_TIMEOUT=600
ENV DEFAULT_MODEL_SERVER_TIMEOUT=600

ENV SAGEMAKER_PROGRAM inference.py

# Label for multi-model support in SageMaker
LABEL com.amazonaws.sagemaker.capabilities.multi-models=true

# Copy the code directory to the Docker image
COPY /code /opt/ml/code

# Install system dependencies and FFmpeg
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir numpy soundfile scipy tqdm librosa demucs onnxruntime-gpu \
    'torch>=1.13.0' pyyaml ml_collections pytorch_lightning 'samplerate==0.1.0' \
    'segmentation_models_pytorch==0.3.3' tqdm

# Freeze the installed Python packages
RUN pip freeze > /opt/ml/code/requirements.txt

# Expose any necessary ports if required
EXPOSE 8080

# Command to run when starting the container
CMD ["python", "/opt/ml/code/inference.py"]
