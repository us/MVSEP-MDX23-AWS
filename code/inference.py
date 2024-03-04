import json
import logging
import os
import tempfile
import boto3
import librosa
import soundfile as sf
import numpy as np
from main import EnsembleDemucsMDXMusicSeparationModel

os.environ['DEFAULT_TS_RESPONSE_TIMEOUT'] = 600
# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for development, INFO for production

def model_fn(model_dir):
    """Load the ensemble model for music separation."""
    logger.info(f"Loading model from directory: {model_dir}")
    model = EnsembleDemucsMDXMusicSeparationModel(model_dir=model_dir)
    return model

def download_file_from_s3(s3_path):
    s3 = boto3.client('s3')
    bucket_name, key = s3_path.replace("s3://", "").split("/", 1)
    temp_dir = tempfile.mkdtemp()
    local_filename = os.path.join(temp_dir, os.path.basename(key))
    logger.info(f"Downloading file from S3: {s3_path} to {local_filename}")
    s3.download_file(bucket_name, key, local_filename)
    return local_filename

def upload_file_to_s3(local_path, bucket_name, s3_path):
    s3 = boto3.client('s3')
    s3.upload_file(local_path, bucket_name, s3_path)
    logger.info(f"Uploaded file to S3: {s3_path}")
    return f"s3://{bucket_name}/{s3_path}"

def input_fn(request_body, request_content_type):
    """Preprocess incoming audio data and additional parameters before prediction."""
    logger.info('Received request_body: %s', request_body[:100])  # Print first 100 chars for debugging
    logger.info('Received request_content_type: %s', request_content_type)
    
    
    # Ensure the correct content type is being used

    # Ensure the correct content type is being used
    if request_content_type != 'application/json':
        logger.error('Unsupported content type: %s', request_content_type)
        raise ValueError(f'Unsupported content type: {request_content_type}')

    try:
        input_data = json.loads(request_body)
        logger.info('Parsed input data successfully.')
        
        
        # Additional debugging to ensure input_data is as expected

        # Additional debugging to ensure input_data is as expected
        if not isinstance(input_data, dict):
            logger.error('Parsed input data is not a dictionary. Actual type: %s', type(input_data))
            raise ValueError('Parsed input data is not a dictionary.')

        # Assuming 's3_audio_path' key contains the S3 path of the audio file
        s3_audio_path = input_data['s3_audio_path']
        local_audio_path = download_file_from_s3(s3_audio_path)
        audio, sample_rate = librosa.load(local_audio_path, sr=44100, mono=False)
        logger.info(f"Loaded audio from {s3_audio_path} with shape: {audio.shape}, sample_rate: {sample_rate}")
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
            logger.info(f"Reshaped audio to: {audio.shape}")
        
        return {
            'audio': audio,
            'sr': sample_rate,
            'options': input_data.get('options', {
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
                                        "output_format": "FLOAT",
                                    })
        }
    except Exception as e:
        logger.error(f"Error processing input data: {e}, request_body: {request_body}, request_content_type: {request_content_type}")
        raise

def predict_fn(input_data, model):
    logger.info('Performing separation on audio data')
    audio, sample_rate, options = input_data['audio'], input_data['sr'], input_data['options']
    logger.info(f"Audio shape: {audio.shape}, sample_rate: {sample_rate}, options: {options}")
    result, sample_rates, instruments = model.separate_music_file(audio.T, sample_rate, options)
    
    return result, sample_rates, instruments, options

def output_fn(prediction, accept='application/json'):
    result, sample_rates, instruments, options = prediction
    bucket_name = options['output_bucket']
    s3_folder_path = 'outputs'
    s3_paths = []
    logger.info(f"Uploading separated audio to S3 bucket: {bucket_name}, folder: {s3_folder_path}")
    for instrum in instruments:
        output_name = f'{instrum}.wav'
        local_path = os.path.join(tempfile.mkdtemp(), output_name)
        sf.write(local_path, result[instrum], sample_rates[instrum], format='WAV')
        s3_path = f"{s3_folder_path}/{output_name}"
        s3_uri = upload_file_to_s3(local_path, bucket_name, s3_path)
        s3_paths.append(s3_uri)
        logger.info(f"Uploaded {instrum} to S3: {s3_uri}")

    if accept == 'application/json':
        return {"out_paths": s3_paths}
    else:
        raise ValueError(f"Unsupported accept header: {accept}")
