import os

from diarization.audio import convert_audio_to_wav
from diarization.diarization import count_speakers

processing = os.getenv("PROCESSING")
dataset = os.getenv("DATASET")

print(dataset, processing)

audio_path = "./dataset/two-men-conversing.mp3"
output_path = "./processing/output.wav"

# convert_audio_to_wav(audio_path)
print(count_speakers(output_path))
