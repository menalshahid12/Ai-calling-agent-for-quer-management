"""
Speech-to-Text via Groq Whisper with post-processing corrections.
"""

import os, re, tempfile

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CORRECTIONS = {
    r"\bees\b": "fees",
    r"\bfree\b(?=\s+(structure|per|semester))": "fee",
    r"\bist\b": "IST",
    r"\bspace technology\b": "Space Technology",
    r"\baerospace\b": "Aerospace",
    r"\bavionics\b": "Avionics",
    r"\bmatric\b": "Matric",
    r"\bfsc\b": "FSc",
    r"\bhostle\b": "hostel",
    r"\bmerit\b": "merit",
}


def transcribe(audio_bytes: bytes, content_type: str = "audio/webm") -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    suffix = ".webm"
    if "wav" in content_type:
        suffix = ".wav"
    elif "ogg" in content_type:
        suffix = ".ogg"
    elif "mp4" in content_type or "m4a" in content_type:
        suffix = ".m4a"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()

    try:
        with open(tmp.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(f"audio{suffix}", audio_file),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )
        text = str(transcription).strip()
    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass

    text = _apply_corrections(text)
    return text


def _apply_corrections(text: str) -> str:
    for pattern, replacement in CORRECTIONS.items():
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
    return text
