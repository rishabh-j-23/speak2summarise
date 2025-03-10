import os
from pydub import AudioSegment

# from pydub.audio_segment import AudioSegment
import noisereduce as nr
import numpy as np


def convert_audio_to_wav(audio_file_path: str, output_file_name="output.wav"):
    """
    Convert an audio file to WAV format and reduce noise.

    Parameters:
        audio_file_path (str): Path to the input audio file.
        output_file_name (str): Name of the output file (default: 'output.wav').
    """
    try:
        # Load audio file
        audio = AudioSegment.from_file(audio_file_path)

        # Convert audio to numpy array
        samples = np.array(
            audio.get_array_of_samples(), dtype=np.float32
        )  # Use float for signal processing
        if audio.channels == 2:
            # Stereo to mono conversion (averaging channels)
            samples = samples.reshape((-1, 2)).mean(axis=1)

        # Normalize signal
        samples = samples / np.max(np.abs(samples))  # Scale between -1 and 1

        # Reduce noise
        reduced_noise = nr.reduce_noise(y=samples, sr=audio.frame_rate)

        # Scale back to original range
        reduced_noise = (reduced_noise * 32767).astype(np.int16)

        # Convert reduced noise signal back to audio
        reduced_audio = AudioSegment(
            reduced_noise.tobytes(),
            frame_rate=audio.frame_rate,
            sample_width=audio.sample_width,
            channels=1,  # Use mono for reduced audio
        )

        # Save reduced audio to file
        processing_path = os.getenv("PROCESSING", ".")  # Default to current directory
        output_path = os.path.join(processing_path, output_file_name)
        reduced_audio.export(output_path, format="wav")

        print(f"Reduced audio saved to: {output_path}")
        return output_path

    except Exception as e:
        print(f"Error processing audio: {e}")
        raise
