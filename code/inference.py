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
logger.setLevel(logging.DEBUG)  # Set to DEBUG for development, INFO for production

# # Default options for model prediction
# default_options = {
#     "overlap_demucs": 0.1,
#     "overlap_VOCFT": 0.1,
#     "overlap_VitLarge": 1,
#     "overlap_InstVoc": 1,
#     "weight_InstVoc": 8g,
#     "weight_VOCFT": 1,
#     "weight_VitLarge": 5,
#     "large_gpu": True,
#     "BigShifts": 7,
#     "vocals_only": False,
#     "use_VOCFT": False,
#     "output_format": "FLOAT",
# }

def model_fn(model_dir):
    """Load the ensemble model for music separation."""
    logger.info(f"Loading model from directory: {model_dir}")
    model = EnsembleDemucsMDXMusicSeparationModel(model_dir=model_dir)
    return model


def input_fn(request_body, request_content_type):
    """Preprocess incoming audio data and additional parameters before prediction."""
    logger.info('Received request_body: %s', request_body[:100])  # Print first 100 chars for debugging
    logger.info('Received request_content_type: %s', request_content_type)
    
    # Ensure the correct content type is being used
    if request_content_type != 'application/json':
        logger.error('Unsupported content type: %s', request_content_type)
        raise ValueError(f'Unsupported content type: {request_content_type}')
    
    try:
        # Parse the JSON body to extract audio data and parameters
        input_data = json.loads(request_body)
        logger.info('Parsed input data successfully.', input_data)
        
        # Additional debugging to ensure input_data is as expected
        if not isinstance(input_data, dict):
            logger.error('Parsed input data is not a dictionary. Actual type: %s', type(input_data))
            raise ValueError('Parsed input data is not a dictionary.')
        
        # Decode the base64-encoded audio data
        audio_data_base64 = input_data['audio']
        audio_data = base64.b64decode(audio_data_base64)
        # audio_buffer = io.BytesIO(audio_data)
        
        # Load the audio with librosa
        audio, sample_rate = librosa.load(audio_data, mono=False, sr=44100)
        if len(audio.shape) == 1:
            audio = np.stack([audio, audio], axis=0)
        
        # Combine audio data and additional parameters in the returned dictionary
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
        logger.error(f"Error processing input data: {e},\nrequest_body: {request_body}, request_content_type: {request_content_type}\n")
        raise


def predict_fn(input_data, model):
    """Run prediction on preprocessed audio data."""
    logger.info('Performing separation on audio data')
    audio, sample_rate, options = input_data['audio'], input_data['sr'], input_data['options']
    result, sample_rates, instruments = model.separate_music_file(audio.T, sample_rate, options)
    
    return result, sample_rates, instruments, options


def output_fn(prediction, accept='application/octet-stream'):
    """
    Process the prediction output to audio buffers and store them in a dictionary.
    """
    result, sample_rates, instruments, options = prediction
    audio_buffers = {}
    
    for instrum in instruments:
        output_name = '_{}.wav'.format(instrum)
        buffer = io.BytesIO()
        sf.write(buffer, result[instrum], sample_rates[instrum], format='WAV')
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    # Instrumental part 1
    if 'instrum' in result:
        inst = result['instrum']
        output_name = '_instrum.wav'
        buffer = io.BytesIO()
        sf.write(buffer, inst, sample_rates.get('instrum', 44100), format='WAV')
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    # Instrumental part 2, if vocals_only option is False
    if not options['vocals_only'] and all(x in result for x in ['bass', 'drums', 'other']):
        inst2 = result['bass'] + result['drums'] + result['other']
        output_name = '_instrum2.wav'
        buffer = io.BytesIO()
        sf.write(buffer, inst2, sample_rates.get('bass', 44100), format='WAV')  # Assuming same SR for all
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    if accept == 'application/octet-stream':
        # This example simply returns the dictionary of buffers
        # Adapt this as needed for your use case, e.g., packaging multiple files, handling accept types
        return audio_buffers
    else:
        raise ValueError(f"Unsupported accept header: {accept}")

