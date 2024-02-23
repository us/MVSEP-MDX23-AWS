import inference  # Import your inference script here
import os
import torchaudio


def main():
    # Set the model directory (adjust this to your model's location)
    model_dir = '../'

    # Load your model with default_options
    # Assuming default_options is accessible within your inference script
    model = inference.model_fn(model_dir)

    # Simulate incoming audio data (replace 'sample_audio.mp3' with your audio file)
    with open('sample_audio.mp3', 'rb') as audio_file:
        audio_data = audio_file.read()

    # Prepare the input data
    input_data = inference.input_fn(audio_data, 'application/octet-stream')

    # Make a prediction
    prediction, sample_rates = inference.predict_fn(input_data, model)

    # Format the output data
    output_data = inference.output_fn((prediction, sample_rates), 'application/json')

    # Print the output data
    print(output_data)

if __name__ == "__main__":
    main()
