import logging
import os
import tempfile
from typing import List
import whisper
import requests
from pydub import AudioSegment

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Constants
SARVAM_PIECE_SECONDS = 25
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

# Global Whisper model lazy-loader
_whisper_model = None


def load_whisper_model():
    """Lazily loads and caches the Whisper model."""
    global _whisper_model
    if _whisper_model is None:
        logging.info(f"Loading Whisper model: {WHISPER_MODEL_NAME}...")
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME)
        logging.info("Whisper model loaded successfully.")
    return _whisper_model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    """Transcribes a local audio chunk using OpenAI's Whisper."""
    model = load_whisper_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result.get("text", "").strip()


def _send_to_sarvam(session: requests.Session, piece_path: str) -> str:
    """Sends a single audio slice (≤30s) to Sarvam AI using a pooled connection."""
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}

        response = session.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=60,
        )

    if not response.ok:
        logging.error(f"Sarvam API error [{response.status_code}]: {response.text}")
        response.raise_for_status()

    return response.json().get("transcript", "").strip()


def transcribe_chunk_sarvam(chunk_path: str) -> str:
    """
    Splits long chunks into 25-second slices to satisfy Sarvam's 30s limit,
    transcribes them via the API, and merges the results.
    """
    if not SARVAM_API_KEY:
        raise ValueError("SARVAM_API_KEY environment variable is not set.")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    transcripts = []

    total_pieces = (len(audio) + piece_ms - 1) // piece_ms

    # Use a persistent session to keep the connection open across slices
    with requests.Session() as session:
        for i, start in enumerate(range(0, len(audio), piece_ms)):
            piece = audio[start : start + piece_ms]

            # Use tempfile to cleanly manage temporary audio slices
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                piece_path = tmp_file.name
                piece.export(piece_path, format="wav")

            try:
                logging.info(f"  → Sarvam processing slice {i + 1}/{total_pieces}...")
                text = _send_to_sarvam(session, piece_path)
                if text:
                    transcripts.append(text)
            finally:
                if os.path.exists(piece_path):
                    os.remove(piece_path)

    return " ".join(transcripts)


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    """Routes the audio chunk to the appropriate engine based on the language."""
    if language.lower() == "hinglish":
        return transcribe_chunk_sarvam(chunk_path)
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: List[str], language: str = "english") -> str:
    """Transcribes a list of audio chunks and returns the combined text."""
    if not chunks:
        logging.warning("No audio chunks provided for transcription.")
        return ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    logging.info(f"Starting batch transcription using engine: {engine}")

    full_transcripts = []
    for i, chunk in enumerate(chunks):
        logging.info(f"Processing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        if text:
            full_transcripts.append(text)

    logging.info("Transcription process complete.")
    return " ".join(full_transcripts)