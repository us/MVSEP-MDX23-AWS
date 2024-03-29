import base64
import json
import inference_colab  # Import your inference script here


def main():
    # Set the model directory (adjust this to your model's location)
    model_dir = '../'
    # cpu_test = False
    # if cpu_test:
    #     test_inference.default_options = {
    #     "overlap_demucs": 0.1,
    #     "overlap_VOCFT": 0.1,
    #     "overlap_VitLarge": 1,
    #     "overlap_InstVoc": 1,
    #     "weight_InstVoc": 8,
    #     "weight_VOCFT": 1,
    #     "weight_VitLarge": 5,
    #     "large_gpu": False,
    #     "BigShifts": 7,
    #     "vocals_only": False,
    #     "use_VOCFT": False,
    #     "output_format": "FLOAT",
    # }
    # Load your model with default_options
    # Assuming default_options is accessible within your inference script
    model = inference_colab.model_fn(model_dir)

    # Simulate incoming audio data (replace 'sample_audio.mp3' with your audio file)
    with open('../sample_audio.mp3', 'rb') as audio_file:
        audio_data = audio_file.read()
    
    
    # Base64-encode the binary audio data
    encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')

    input_json = json.dumps({
        "audio": '../sample_audio.mp3',
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
            "output_format": "FLOAT",
        }
    })

    # Prepare the input data
    input_data = inference_colab.input_fn(input_json, 'application/json')

    # Make a prediction
    prediction, sample_rates, instruments, options = inference_colab.predict_fn(input_data, model)

    # Format the output data
    output_data = inference_colab.output_fn((prediction, sample_rates, instruments, options), 'application/octet-stream')

    # Print the output data
    print(output_data)

if __name__ == "__main__":
    main()
