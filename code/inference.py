import logging

import librosa
import numpy as np
from main import EnsembleDemucsMDXMusicSeparationModel
import torchaudio
import torch
import io
import base64

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Set to DEBUG for development, INFO for production

# Default options for model prediction
default_options = {
    # "cpu": False,
    "overlap_demucs": 0.1,
    "overlap_VOCFT": 0.1,
    "overlap_VitLarge": 1,
    "overlap_InstVoc": 1,
    "weight_InstVoc": 8,
    "weight_VOCFT": 1,
    "weight_VitLarge": 5,
    "single_onnx": False,
    "large_gpu": True,
    "BigShifts": 7,
    "vocals_only": False,
    "use_VOCFT": False,
    "output_format": "FLOAT",
}

def model_fn(model_dir):
    """Load the ensemble model for music separation."""
    logger.info(f"Loading model from directory: {model_dir}")
    model = EnsembleDemucsMDXMusicSeparationModel(model_dir=model_dir, options=default_options)
    return model

def input_fn(request_body, request_content_type):
    """Preprocess incoming audio data before prediction."""
    logger.info('Processing input data')
    if request_content_type != 'application/octet-stream':
        raise ValueError(f'Unsupported content type: {request_content_type}')
    
    try:
        audio_buffer = io.BytesIO(request_body)
        audio, sample_rate = librosa.load(audio_buffer, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        # # waveform, sample_rate = torchaudio.load(audio_buffer)
        # if waveform.size(0) == 1:
        #     waveform = waveform.repeat(2, 1)  # Ensure stereo audio
        return {'audio': audio, 'sr': sample_rate, 'index': 0, 'total': 1}
    except Exception as e:
        logger.error(f"Error processing audio with torchaudio: {e}")
        raise

def predict_fn(input_data, model):
    """Run prediction on preprocessed audio data."""
    logger.info('Performing separation on audio data')
    audio, sample_rate = input_data['audio'], input_data['sr']
    result, sample_rates = model.separate_music_file(audio.T, sample_rate, input_data['index'], input_data['total'])
    return result, sample_rates

def output_fn(prediction, content_type):
    """Postprocess and format the prediction output."""
    logger.info('Formatting output data')
    result, sample_rates = prediction
    output_responses = []

    for instrum, audio_data in result.items():
        audio_tensor = audio_data if isinstance(audio_data, torch.Tensor) else torch.tensor(audio_data, dtype=torch.float32)
        audio_tensor = audio_tensor.to('cpu')  # Ensure data is on CPU for serialization

        buffer = io.BytesIO()
        # Specify the format explicitly, assuming WAV format
        torchaudio.save(buffer, audio_tensor.unsqueeze(0), sample_rate=sample_rates[instrum], format="wav")
        buffer.seek(0)
        encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')  # Encode for transmission

        output_responses.append({'instrument': instrum, 'audio_data': encoded_audio})
        logger.info(f'Output processed for instrument: {instrum}')

    return output_responses
"""
Processing vocals: DONE!
Starting Demucs processing...
Processing with htdemucs_ft...
Traceback (most recent call last):
  File "/home/ec2-user/SageMaker/MVSEP-MDX23-AWS/code/test_inference.py", line 48, in <module>
    main()
  File "/home/ec2-user/SageMaker/MVSEP-MDX23-AWS/code/test_inference.py", line 39, in main
    prediction, sample_rates = inference.predict_fn(input_data, model)
  File "/home/ec2-user/SageMaker/MVSEP-MDX23-AWS/code/inference.py", line 62, in predict_fn
    result, sample_rates = model.separate_music_file(audio.T, sample_rate, input_data['index'], input_data['total'])
  File "/home/ec2-user/SageMaker/MVSEP-MDX23-AWS/code/main.py", line 598, in separate_music_file
    print('Processing with htdemucs...', repo=pathlib.Path('..'))
TypeError: 'repo' is an invalid keyword argument for print()
"""