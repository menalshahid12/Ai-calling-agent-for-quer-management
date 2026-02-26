"""
LLM answer generation via Groq, with IST-specific system prompt.
"""

import os, re

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are an official AI voice assistant for the Institute of Space Technology (IST), Islamabad, Pakistan.
You answer admission-related questions for callers.

STRICT RULES:
1. Answer ONLY from the provided IST context below. Never invent facts, dates, figures, or names.
2. Keep answers short: 1-2 sentences (up to 4 for complex multi-part answers).
3. Do NOT start your reply by repeating the user's question. Start directly with the answer.
4. Present information confidently as factual from IST. If the user says "you're wrong", politely restate the same facts from context (e.g. "According to IST's official information, ...").
5. For yes/no questions, answer "Yes" or "No" first, then a brief supporting line from context.
6. When stating fees, always use "lakh" and "thousand" (Pakistani convention). Never say "million" for fees.
7. If the context has relevant information, ALWAYS answer from it. Never say "I don't have information" or "technical issue" when the context covers the topic.
8. If the question is truly NOT answerable from the context and is NOT a simple yes/no:
   - Say EXACTLY: "I will forward your query to the IST admissions office. Could you please provide your phone number so they can call you back?"
   - Do NOT make up an answer.
9. Never say "technical issue", "technical difficulties", or "I'm having trouble" to the user.
10. Never tell the user to "check the website" or "call the office" as the main response when the answer is in context.
11. You support long conversations (10+ questions). Use the recent conversation history for follow-up questions.
12. When asked for aggregate calculation, compute it using the formula from context and show the result.
"""

ESCALATION_MSG = "I will forward your query to the IST admissions office. Could you please provide your phone number so they can call you back?"

TECHNICAL_ISSUE_PATTERNS = [
    r"technical\s+(issue|difficult|problem|error)",
    r"i('m|\s+am)\s+(having|experiencing)\s+(a\s+)?technical",
    r"i\s+don'?t\s+have\s+(that\s+)?information",
    r"i\s+cannot\s+(find|access|retrieve)",
    r"unable\s+to\s+(find|access|provide)",
]


def generate_answer(user_query: str, context: str, conversation_history: list[dict]) -> str:
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)

    recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in recent_history:
        messages.append({"role": "user", "content": turn.get("user", "")})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})

    user_content = f"""Context from IST knowledge base:
---
{context}
---

Caller's question: {user_query}

Answer the question using ONLY the context above. Be concise and direct."""

    messages.append({"role": "user", "content": user_content})

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=300,
        )
        answer = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[LLM] Error: {e}")
        answer = ESCALATION_MSG

    answer = _sanitize_answer(answer)
    return answer


def _sanitize_answer(answer: str) -> str:
    for pattern in TECHNICAL_ISSUE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return ESCALATION_MSG
    return answer
