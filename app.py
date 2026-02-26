"""
IST AI Calling Agent — Flask backend.
Handles call sessions, STT → RAG → LLM → TTS pipeline.
"""

import os, json, re, uuid, time, threading
from datetime import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import knowledge_base as kb
import stt
import llm
import tts

app = Flask(__name__, static_folder="static")
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024
CORS(app)

LOGS_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)

LEAD_LOG = os.path.join(LOGS_DIR, "lead_logs.txt")
CALL_RECORDS = os.path.join(LOGS_DIR, "call_records.json")

_sessions: dict[str, dict] = {}
_log_lock = threading.Lock()

GREETING = "Hello, this is the Institute of Space Technology. How can I help you with admissions today?"

END_PHRASES = [
    "no more questions", "end call", "goodbye", "good bye", "bye",
    "that's all", "thats all", "thank you bye", "nothing else",
    "i'm done", "im done", "hang up", "end the call", "no thanks",
]

PHONE_PATTERN = re.compile(r"(?:\+?92|0)?[\s\-]?3\d{2}[\s\-]?\d{7}")


def _get_session(sid: str) -> dict:
    if sid not in _sessions:
        _sessions[sid] = {
            "id": sid,
            "history": [],
            "start_time": datetime.now().isoformat(),
            "escalated": False,
            "awaiting_phone": False,
            "phone_number": None,
            "turns": 0,
        }
    return _sessions[sid]


def _save_call_record(session: dict):
    session["end_time"] = datetime.now().isoformat()
    record = {
        "call_id": session["id"],
        "start_time": session["start_time"],
        "end_time": session["end_time"],
        "turns": session["turns"],
        "escalated": session["escalated"],
        "phone_number": session.get("phone_number"),
        "history": session["history"],
    }
    with _log_lock:
        records = []
        if os.path.exists(CALL_RECORDS):
            try:
                with open(CALL_RECORDS, "r") as f:
                    records = json.load(f)
            except Exception:
                records = []
        records.append(record)
        with open(CALL_RECORDS, "w") as f:
            json.dump(records, f, indent=2, default=str)


def _save_lead(phone: str, session: dict):
    last_query = ""
    if session["history"]:
        last_query = session["history"][-1].get("user", "")
    line = f"{datetime.now().isoformat()} | phone={phone} | call_id={session['id']} | query={last_query}\n"
    with _log_lock:
        with open(LEAD_LOG, "a") as f:
            f.write(line)


def _is_end_call(text: str) -> bool:
    text_lower = text.lower().strip()
    return any(ep in text_lower for ep in END_PHRASES)


def _extract_phone(text: str) -> str | None:
    m = PHONE_PATTERN.search(text)
    if m:
        return re.sub(r"[\s\-]", "", m.group())
    digits = re.sub(r"[^\d]", "", text)
    if len(digits) >= 10 and digits[-10:-7] in ("30", "31", "32", "33", "34", "35", "36", "37", "38", "39"):
        return digits[-11:] if len(digits) >= 11 else "0" + digits[-10:]
    return None


@app.route("/")
def index():
    return send_from_directory("static", "index.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/greeting", methods=["GET"])
def greeting():
    session_id = request.args.get("session_id", uuid.uuid4().hex)
    _get_session(session_id)
    audio_url = tts.synthesize(GREETING, session_id)
    return jsonify({
        "session_id": session_id,
        "text": GREETING,
        "audio_url": audio_url,
    })


@app.route("/api/query", methods=["POST"])
def query():
    start = time.time()
    try:
        audio_file = request.files.get("audio")
        session_id = request.form.get("session_id", uuid.uuid4().hex)
        session = _get_session(session_id)

        if not audio_file:
            print("[Query] No audio file in request")
            return jsonify({"error": "No audio file provided"}), 400

        audio_bytes = audio_file.read()
        content_type = audio_file.content_type or "audio/webm"
        print(f"[Query] Session={session_id}, audio={len(audio_bytes)} bytes, type={content_type}")

        user_text = stt.transcribe(audio_bytes, content_type)
        print(f"[Query] STT result: '{user_text}'")

        if not user_text or len(user_text.strip()) < 2:
            print("[Query] STT returned empty/short text, skipping")
            return jsonify({"error": "Could not understand audio", "text": ""}), 200

        if _is_end_call(user_text):
            goodbye = "Thank you for calling the Institute of Space Technology. Goodbye!"
            audio_url = tts.synthesize(goodbye, session_id)
            session["history"].append({"user": user_text, "assistant": goodbye})
            session["turns"] += 1
            _save_call_record(session)
            return jsonify({
                "text": goodbye,
                "user_text": user_text,
                "audio_url": audio_url,
                "end_call": True,
                "turns": session["turns"],
                "duration": round(time.time() - _parse_start(session), 1),
            })

        if session.get("awaiting_phone"):
            phone = _extract_phone(user_text)
            if phone:
                session["phone_number"] = phone
                session["awaiting_phone"] = False
                _save_lead(phone, session)
                reply = "Thank you! I have noted your number. The IST admissions office will call you back. Is there anything else I can help you with?"
            else:
                reply = "I could not catch your phone number. Could you please say it again slowly? For example, zero three zero zero, one two three four five six seven."
            audio_url = tts.synthesize(reply, session_id)
            session["history"].append({"user": user_text, "assistant": reply})
            session["turns"] += 1
            return jsonify({
                "text": reply,
                "user_text": user_text,
                "audio_url": audio_url,
                "end_call": False,
            })

        prev_user = ""
        if session["history"]:
            prev_user = session["history"][-1].get("user", "")

        search_query = user_text
        if len(user_text.split()) <= 4 and prev_user:
            search_query = f"{prev_user} {user_text}"

        context = kb.search(search_query)
        reply = llm.generate_answer(user_text, context, session["history"])

        if reply == llm.ESCALATION_MSG:
            session["escalated"] = True
            session["awaiting_phone"] = True

        audio_url = tts.synthesize(reply, session_id)
        session["history"].append({"user": user_text, "assistant": reply})
        session["turns"] += 1

        latency = round(time.time() - start, 2)
        print(f"[Turn {session['turns']}] {latency}s | Q: {user_text[:60]} | A: {reply[:60]}")

        return jsonify({
            "text": reply,
            "user_text": user_text,
            "audio_url": audio_url,
            "end_call": False,
            "latency": latency,
        })

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[Query] UNHANDLED ERROR: {e}")
        return jsonify({"error": str(e), "text": ""}), 500


def _parse_start(session: dict) -> float:
    try:
        return datetime.fromisoformat(session["start_time"]).timestamp()
    except Exception:
        return time.time()


def _periodic_cleanup():
    while True:
        time.sleep(300)
        tts.cleanup_old_audio(600)

cleanup_thread = threading.Thread(target=_periodic_cleanup, daemon=True)
cleanup_thread.start()


with app.app_context():
    kb.init_kb()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
