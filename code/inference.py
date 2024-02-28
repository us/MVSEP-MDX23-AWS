import base64
import json
import logging
import librosa
import numpy as np
from main import EnsembleDemucsMDXMusicSeparationModel
import io
import soundfile as sf

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust logging level as needed

def model_fn(model_dir):
    """Load the ensemble model for music separation."""
    logger.info(f"Loading model from directory: {model_dir}")
    model = EnsembleDemucsMDXMusicSeparationModel(model_dir=model_dir)
    return model

def transform_fn(model, request_body, request_content_type, response_content_type):
    """
    Function to preprocess request, perform prediction, and postprocess before sending response.
    """
    logger.info('Transforming input for prediction')
    
    # Preprocessing
    if request_content_type != 'application/json':
        logger.error('Unsupported content type: %s', request_content_type)
        raise ValueError(f'Unsupported content type: {request_content_type}')

    try:
        # Parse JSON body to extract audio data and parameters
        input_data = json.loads(io.BytesIO(request_body))
        audio_data_base64 = input_data['audio']
        audio_data = base64.b64decode(audio_data_base64)
        audio_buffer = io.BytesIO(audio_data)
        
        # Load audio with librosa
        audio, sample_rate = librosa.load(audio_buffer, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)

        # Combine audio data and additional parameters
        preprocessed_data = {
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
            })
        }
    except Exception as e:
        logger.error(f"Error processing input data: {e}")
        raise

    # Prediction
    logger.info('Performing prediction')
    result, sample_rates, instruments = model.separate_music_file(preprocessed_data['audio'].T, preprocessed_data['sr'], preprocessed_data['options'])
    
    # Postprocessing
    logger.info('Postprocessing prediction output')
    audio_buffers = {}
    for instrum in instruments:
        output_name = f'_{instrum}.wav'
        buffer = io.BytesIO()
        sf.write(buffer, result[instrum], sample_rates[instrum], format='WAV')
        buffer.seek(0)
        audio_buffers[output_name] = buffer.getvalue()  # Modified to getvalue for binary data

    # Example for one instrument, similar logic for others as needed
    # Adapt this part based on how you want to package and return the audio buffers
    
    if response_content_type == 'application/octet-stream':
        # Assuming returning binary data for one instrument for simplicity
        return audio_buffers[next(iter(audio_buffers))], response_content_type
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")

