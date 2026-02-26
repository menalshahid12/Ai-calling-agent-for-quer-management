"""
Speech-to-Text via Groq Whisper with post-processing corrections.
"""

import os, re, tempfile, time

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

CORRECTIONS = {
    r"\bees\b": "fees",
    r"\bfree\b(?=\s+(structure|per|semester))": "fee",
    r"\bisd\b": "IST",
    r"\bist\b": "IST",
    r"\bi\.s\.t\.?\b": "IST",
    r"\biced\b": "IST",
    r"\bisty\b": "IST",
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

    if not GROQ_API_KEY:
        print("[STT] ERROR: GROQ_API_KEY is not set!")
        return ""

    client = Groq(api_key=GROQ_API_KEY)

    print(f"[STT] Received {len(audio_bytes)} bytes, type={content_type}")

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
        start = time.time()
        with open(tmp.name, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=(f"audio{suffix}", audio_file),
                model="whisper-large-v3-turbo",
                language="en",
                response_format="text",
            )
        text = str(transcription).strip()
        elapsed = round(time.time() - start, 2)
        print(f"[STT] Transcribed in {elapsed}s: '{text}'")
    except Exception as e:
        print(f"[STT] ERROR: {e}")
        return ""
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
