import logging
import os
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
    "cpu": True,
    "overlap_demucs": 0.1,
    "overlap_VOCFT": 0.1,
    "overlap_VitLarge": 1,
    "overlap_InstVoc": 1,
    "weight_InstVoc": 8,
    "weight_VOCFT": 1,
    "weight_VitLarge": 5,
    "single_onnx": False,
    "large_gpu": False,
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
        waveform, sample_rate = torchaudio.load(audio_buffer)
        if waveform.size(0) == 1:
            waveform = waveform.repeat(2, 1)  # Ensure stereo audio
        return {'audio': waveform, 'sr': sample_rate, 'index': 0, 'total': 1}
    except Exception as e:
        logger.error(f"Error processing audio with torchaudio: {e}")
        raise

def predict_fn(input_data, model):
    """Run prediction on preprocessed audio data."""
    logger.info('Performing separation on audio data')
    waveform, sample_rate = input_data['audio'], input_data['sr']
    result, sample_rates = model.separate_music_file(waveform.T, sample_rate, input_data['index'], input_data['total'])
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
        torchaudio.save(buffer, audio_tensor, sample_rates[instrum])
        buffer.seek(0)
        encoded_audio = base64.b64encode(buffer.read()).decode('utf-8')  # Encode for transmission
        
        output_responses.append({'instrument': instrum, 'audio_data': encoded_audio})
        logger.info(f'Output processed for instrument: {instrum}')
    
    return output_responses
