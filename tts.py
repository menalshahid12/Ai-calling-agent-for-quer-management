"""
Text-to-Speech via Edge TTS. Each call produces a unique file so
concurrent sessions never overwrite each other.
"""

import os, uuid, asyncio

AUDIO_DIR = os.path.join(os.path.dirname(__file__), "static", "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

VOICE = "en-US-JennyNeural"
RATE = "+10%"


async def _synthesize(text: str, filepath: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
    await communicate.save(filepath)


def synthesize(text: str, session_id: str = "") -> str:
    filename = f"{session_id}_{uuid.uuid4().hex[:8]}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    loop = None
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        pass

    if loop and loop.is_running():
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as pool:
            future = pool.submit(asyncio.run, _synthesize(text, filepath))
            future.result()
    else:
        asyncio.run(_synthesize(text, filepath))

    return f"/static/audio/{filename}"


def cleanup_old_audio(max_age_seconds: int = 600):
    import time
    now = time.time()
    for fname in os.listdir(AUDIO_DIR):
        fpath = os.path.join(AUDIO_DIR, fname)
        try:
            if now - os.path.getmtime(fpath) > max_age_seconds:
                os.unlink(fpath)
        except Exception:
            pass
