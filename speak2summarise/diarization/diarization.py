import os
from dotenv import load_dotenv

from collections import defaultdict
from pyannote.audio import Pipeline

load_dotenv()


def count_speakers(audio_path: str):
    """Detect number of speakers in an audio file."""
    # Load pre-trained speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )

    # Apply the pipeline to the audio file
    diarization = pipeline(audio_path)

    # Collect speaker labels
    speakers = defaultdict(list)
    for segment, track, label in diarization.itertracks(yield_label=True):
        speakers[label].append(segment)

    # Count unique speakers
    num_speakers = len(speakers)
    return num_speakers


def segment_audio_by_speaker(audio_path):
    """Segment audio by speaker and provide start and end times."""
    # Load the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=os.getenv("HF_TOKEN"),
    )

    # Apply the pipeline to the audio file
    diarization = pipeline(audio_path)

    # Prepare segments with speaker labels
    segments = []
    for segment, track, label in diarization.itertracks(yield_label=True):
        # Add only segments longer than a threshold to avoid noise artifacts
        if segment.end - segment.start > 0.5:  # Filter short segments
            segments.append(
                {"speaker": label, "start_sec": segment.start, "end_sec": segment.end}
            )

    # Merge adjacent segments with the same speaker
    merged_segments = []
    for seg in segments:
        if merged_segments and merged_segments[-1]["speaker"] == seg["speaker"]:
            merged_segments[-1]["end_sec"] = seg["end_sec"]  # Merge segments
        else:
            merged_segments.append(seg)

    return merged_segments


def transcribe_segments(audio_path, segments, model):
    """Transcribe each audio segment and add text to segments."""
    from pydub import AudioSegment

    audio = AudioSegment.from_wav(audio_path)

    for segment in segments:
        # Extract segment audio
        start_ms = segment["start_sec"] * 1000  # Convert to milliseconds
        end_ms = segment["end_sec"] * 1000  # Convert to milliseconds
        segment_audio = audio[start_ms:end_ms]

        # Save segment to temporary file
        temp_file = "temp_segment.wav"
        segment_audio.export(temp_file, format="wav")

        # Transcribe audio segment
        result = model.transcribe(temp_file)

        # Debugging: Print the result structure
        print(f"Transcription Result: {result}")

        # Safely add transcription to the segment
        segment["text"] = result["segments"][0]["text"]

    return segments
