import runpod
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

def init_model():
    global MODEL
    if MODEL is None:
        model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
        MODEL = model_fn(model_dir)
    return MODEL

def download_file_from_s3(client, s3_path):
    bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
    temp_dir = tempfile.mkdtemp()
    local_filename = os.path.join(temp_dir, os.path.basename(key))
    logger.info(f"Downloading file from S3: {s3_path} to {local_filename}")
    client.download_file(bucket_name, key, local_filename)
    return local_filename

def upload_file_to_s3(client, local_path, bucket_name, s3_path):
    client.upload_file(local_path, bucket_name, s3_path)
    logger.info(f"Uploaded file to S3: {s3_path}")
    return f"s3://{bucket_name}/{s3_path}"

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
        s3_client = boto3.client('s3',
                                 aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
                                 aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
                                 region_name=os.getenv("AWS_REGION"))
        # Initialize model if not already done
        model = init_model()
        
        # Get input data
        input_data = event["input"]
        audio_url = input_data["audio_url"]
        job_id = event["id"]
        
        # Get S3 output settings if provided
        output_bucket = input_data.get("output_bucket")
        output_prefix = input_data.get("output_prefix", "")
        
        # Download audio file from S3
        local_audio_path = download_file_from_s3(s3_client, audio_url)
        
        # Load audio file
        audio, sample_rate = sf.read(local_audio_path)
        os.unlink(local_audio_path)  # Clean up temp file
        
        # Ensure stereo format
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        elif len(audio.shape) == 2 and audio.shape[0] > 2:
            audio = audio.T
        
        # Get options
        # try to get options from input_data if not provided, use default options, log it
        options = input_data.get("options", None)
        if options is None:
            print("No options provided, using default options")
            options = {
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
        print(f"Using options: {options}")

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
            "s3": {}  # Only using S3 storage now
        }
        
        for instrum in instruments:
            # Create temporary file for the separated audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                sf.write(temp_file.name, result[instrum], sample_rates[instrum], format='WAV')
                
                # Upload to S3 if configured
                if output_bucket:
                    s3_key = f"{output_prefix}/{job_id}/{instrum}.wav" if output_prefix else f"{job_id}/{instrum}.wav"
                    output_urls["s3"][instrum] = upload_file_to_s3(s3_client, temp_file.name, output_bucket, s3_key)
                
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