import base64
import json
import inference  # Import your inference script here


def main():
    # Set the model directory (adjust this to your model's location)
    model_dir = '../'
    cpu_test = False
    if cpu_test:
        inference.default_options = {
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
    # Load your model with default_options
    # Assuming default_options is accessible within your inference script
    model = inference.model_fn(model_dir)

    # Simulate incoming audio data (replace 'sample_audio.mp3' with your audio file)
    with open('../sample_audio.mp3', 'rb') as audio_file:
        audio_data = audio_file.read()
    
    
    # Base64-encode the binary audio data
    encoded_audio_data = base64.b64encode(audio_data).decode('utf-8')

    input_json = json.dumps({
        "audio": encoded_audio_data,
        "options": {"cpu": cpu_test}  # Assuming cpu_test is a variable or replace with actual value
    })

    # Prepare the input data
    input_data = inference.input_fn(input_json, 'application/json')

    # Make a prediction
    prediction, sample_rates, instruments = inference.predict_fn(input_data, model)

    # Format the output data
    output_data = inference.output_fn((prediction, sample_rates, instruments), 'application/octet-stream')

    # Print the output data
    print(output_data)

if __name__ == "__main__":
    main()
