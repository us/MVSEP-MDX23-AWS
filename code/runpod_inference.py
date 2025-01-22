import runpod
from runpod.serverless.utils import rp_upload, rp_download
import os
import json
import torch
import logging
from inference_colab import model_fn, predict_fn
import soundfile as sf
import io
import numpy as np
import tempfile
import boto3
from urllib.parse import urlparse

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model
MODEL = None

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=os.environ.get('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'),
    region_name=os.environ.get('AWS_REGION', 'us-east-1')
)

def init_model():
    global MODEL
    if MODEL is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        MODEL = model_fn(model_dir)
    return MODEL

def upload_to_s3(file_path, bucket, key):
    """Upload file to S3 and return S3 URI."""
    try:
        s3_client.upload_file(file_path, bucket, key)
        return f"s3://{bucket}/{key}"
    except Exception as e:
        logger.error(f"Error uploading to S3: {str(e)}")
        raise

def handler(event):
    """
    RunPod handler function for audio separation.
    Expects input in the format:
    {
        "input": {
            "audio_url": "https://bucket-name.s3.region.amazonaws.com/path/to/input.mp3",
            "output_bucket": "output-bucket-name",  # Optional: S3 bucket for persistent storage
            "output_prefix": "path/to/output/folder",  # Optional: S3 prefix for output files
            "options": {
                "overlap_demucs": 0.1,
                "overlap_VOCFT": 0.1,
                "overlap_VitLarge": 1,
                "overlap_InstVoc": 1,
                "weight_InstVoc": 8,
                "weight_VOCFT": 1,
                "weight_VitLarge": 5,
                "large_gpu": True,
                "BigShifts": 7,
                "vocals_only": False,
                "use_VOCFT": False,
                "output_format": "FLOAT"
            }
        }
    }
    """
    try:
        # Initialize model if not already done
        model = init_model()
        
        # Get input data
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        job_id = event["id"]  # Get the job ID for unique file naming
        
        # Get S3 output settings if provided
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        
        # Download audio file using RunPod's utility
        local_audio_path = rp_download.download_file_from_url(audio_url)
        
        # Load audio file
        audio, sample_rate = sf.read(local_audio_path)
        os.unlink(local_audio_path)  # Clean up temp file
        
        # Ensure stereo format
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        elif len(audio.shape) == 2 and audio.shape[0] > 2:
            audio = audio.T
        
        # Get options
        options = input_data.get("options", {
            "overlap_demucs": 0.1,
            "overlap_VOCFT": 0.1,
            "overlap_VitLarge": 1,
            "overlap_InstVoc": 1,
            "weight_InstVoc": 8,
            "weight_VOCFT": 1,
            "weight_VitLarge": 5,
            "large_gpu": True,
            "BigShifts": 7,
            "vocals_only": False,
            "use_VOCFT": False,
            "output_format": "FLOAT"
        })
        
        # Prepare input for prediction
        model_input = {
            'audio': audio,
            'sr': sample_rate,
            'options': options
        }
        
        # Run prediction
        result, sample_rates, instruments, _ = predict_fn(model_input, model)
        
        # Upload results
        output_urls = {
            "runpod": {},  # RunPod temporary storage URLs
            "s3": {}       # S3 permanent storage URLs (if configured)
        }
        
        for instrum in instruments:
            # Create temporary file for the separated audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, result[instrum], sample_rates[instrum], format='WAV')
                
                # Upload to RunPod storage
                output_urls["runpod"][instrum] = rp_upload.upload_file(
                    job_id,
                    temp_file.name,
                    f"{instrum}.wav"
                )
                
                # Upload to S3 if configured
                if output_bucket:
                    s3_key = f"{output_prefix}/{job_id}/{instrum}.wav" if output_prefix else f"{job_id}/{instrum}.wav"
                    output_urls["s3"][instrum] = upload_to_s3(temp_file.name, output_bucket, s3_key)
                
                # Clean up temp file
                os.unlink(temp_file.name)
            
        # Return the results
        return {
            "output": output_urls,
            "sample_rates": {k: int(v) for k, v in sample_rates.items()},
            "instruments": list(instruments)
        }
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return {"error": str(e)}

runpod.serverless.start({"handler": handler}) 