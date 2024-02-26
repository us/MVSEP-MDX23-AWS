import logging

import librosa
import numpy as np
from main import EnsembleDemucsMDXMusicSeparationModel

import io
import soundfile as sf

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
    result, sample_rates, instruments = model.separate_music_file(audio.T, sample_rate, input_data['index'], input_data['total'])
    
    return result, sample_rates, instruments


def output_fn(prediction, accept='application/octet-stream'):
    """
    Process the prediction output to audio buffers and store them in a dictionary.
    """
    result, sample_rates, instruments = prediction
    audio_buffers = {}
    
    for instrum in instruments:
        output_name = '_{}.wav'.format(instrum)
        buffer = io.BytesIO()
        sf.write(buffer, result[instrum], sample_rates[instrum], format='WAV', subtype=default_options.output_format)
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    # Instrumental part 1
    if 'instrum' in result:
        inst = result['instrum']
        output_name = '_instrum.wav'
        buffer = io.BytesIO()
        sf.write(buffer, inst, sample_rates.get('instrum', 44100), format='WAV', subtype=default_options.output_format)
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    # Instrumental part 2, if vocals_only option is False
    if not default_options['vocals_only'] and all(x in result for x in ['bass', 'drums', 'other']):
        inst2 = result['bass'] + result['drums'] + result['other']
        output_name = '_instrum2.wav'
        buffer = io.BytesIO()
        sf.write(buffer, inst2, sample_rates.get('bass', 44100), format='WAV', subtype=default_options.output_format)  # Assuming same SR for all
        buffer.seek(0)
        audio_buffers[output_name] = buffer

    if accept == 'application/octet-stream':
        # This example simply returns the dictionary of buffers
        # Adapt this as needed for your use case, e.g., packaging multiple files, handling accept types
        return audio_buffers
    else:
        raise ValueError(f"Unsupported accept header: {accept}")

