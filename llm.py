"""
LLM answer generation via Groq, with IST-specific system prompt.
"""

import os, re, time

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are the official AI phone assistant for Institute of Space Technology (IST), Islamabad, Pakistan.
You handle admissions queries from callers using ONLY the IST knowledge base context provided to you.

ABSOLUTE RULES — NEVER BREAK THESE:

1. ANSWER FROM CONTEXT: The context below contains verified IST information. If the context mentions fees, programs, departments, merit, hostel, transport, eligibility, deadlines, or any IST topic — you MUST answer from it. Do NOT say you cannot help when the answer is right there in the context.

2. NEVER ESCALATE WHEN CONTEXT HAS THE ANSWER: Only use the escalation phrase when the context has ZERO relevant information for the question. If you see ANY relevant data in the context, answer from it.

3. SHORT AND DIRECT: 1-3 sentences. Start with the answer, never repeat the question back.

4. CONFIDENT: State facts from context as official IST information. If user says "you're wrong", say "According to IST's official information, ..." and repeat the same facts.

5. FEES IN LAKH/THOUSAND: Always say fees in Pakistani convention (lakh and thousand). Never say "million".

6. YES/NO FIRST: For yes/no questions, say "Yes" or "No" first, then one supporting line.

7. NEVER SAY: "technical issue", "technical difficulties", "I don't have information", "I cannot find", "unable to provide". These phrases are FORBIDDEN.

8. NEVER INVENT: Do not make up numbers, dates, names, or facts not in the context.

9. ESCALATION — USE ONLY AS LAST RESORT: If and ONLY if the question has absolutely nothing to do with IST or the context has zero relevant information:
   Say exactly: "I will forward your query to the IST admissions office. Could you please provide your phone number so they can call you back?"

10. AGGREGATE CALCULATION: When asked to calculate aggregate, use the formula from context and compute it.

11. FOLLOW-UPS: Use conversation history for context. "What about fees for that?" refers to the previously discussed program.

REMEMBER: You have the IST knowledge base. USE IT. Answer the question."""

ESCALATION_MSG = "I will forward your query to the IST admissions office. Could you please provide your phone number so they can call you back?"

TECHNICAL_ISSUE_PATTERNS = [
    r"technical\s+(issue|difficult|problem|error)",
    r"i('m|\s+am)\s+(having|experiencing)\s+(a\s+)?technical",
    r"unable\s+to\s+(find|access|provide|assist)",
]

UNNECESSARY_ESCALATION_PATTERNS = [
    r"i\s+don'?t\s+have\s+(that\s+)?(specific\s+)?information",
    r"i\s+(cannot|can'?t)\s+(find|access|retrieve|provide)",
    r"not\s+available\s+in\s+(my|the)\s+(knowledge|context|data)",
    r"beyond\s+(my|the)\s+(scope|knowledge|context)",
]


def _context_has_relevant_info(query: str, context: str) -> bool:
    if not context or context == "No relevant information found.":
        return False
    q_lower = query.lower()
    relevance_words = ["fee", "program", "department", "merit", "aggregate", "hostel",
                       "transport", "admission", "eligib", "deadline", "scholarship",
                       "semester", "lakh", "thousand", "engineering", "computer",
                       "aerospace", "electrical", "avionics", "mechanical", "science",
                       "apply", "entry test", "nat", "bs ", "ms ", "phd"]
    for word in relevance_words:
        if word in q_lower and word in context.lower():
            return True
    return False


def generate_answer(user_query: str, context: str, conversation_history: list[dict]) -> str:
    from groq import Groq

    if not GROQ_API_KEY:
        print("[LLM] ERROR: GROQ_API_KEY not set!")
        return ESCALATION_MSG

    client = Groq(api_key=GROQ_API_KEY)

    recent_history = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for turn in recent_history:
        messages.append({"role": "user", "content": turn.get("user", "")})
        if turn.get("assistant"):
            messages.append({"role": "assistant", "content": turn["assistant"]})

    user_content = f"""Here is the IST knowledge base context for this question:

=== IST CONTEXT START ===
{context}
=== IST CONTEXT END ===

Caller's question: {user_query}

IMPORTANT: The context above contains the answer. Read it carefully and answer directly. Do NOT escalate or forward the query — the information IS in the context above."""

    messages.append({"role": "user", "content": user_content})

    has_context = _context_has_relevant_info(user_query, context)
    max_attempts = 2 if has_context else 1

    for attempt in range(max_attempts):
        try:
            start = time.time()
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                temperature=0.1 + (attempt * 0.15),
                max_tokens=300,
            )
            answer = response.choices[0].message.content.strip()
            elapsed = round(time.time() - start, 2)
            print(f"[LLM] Attempt {attempt+1}, {elapsed}s: '{answer[:100]}'")
        except Exception as e:
            print(f"[LLM] Error: {e}")
            answer = ESCALATION_MSG

        answer = _sanitize_answer(answer, context)

        is_escalation = (answer == ESCALATION_MSG or
                         "forward" in answer.lower() and "phone" in answer.lower())

        if is_escalation and has_context and attempt < max_attempts - 1:
            print(f"[LLM] Escalated despite context existing — retrying (attempt {attempt+2})")
            if messages[-1]["role"] == "user":
                messages[-1]["content"] += "\n\nYou MUST answer from the context. The answer IS there. Do NOT ask for a phone number. Just answer the question directly."
            continue

        return answer

    return answer


def _sanitize_answer(answer: str, context: str = "") -> str:
    for pattern in TECHNICAL_ISSUE_PATTERNS:
        if re.search(pattern, answer, re.IGNORECASE):
            return ESCALATION_MSG

    if context and context != "No relevant information found.":
        for pattern in UNNECESSARY_ESCALATION_PATTERNS:
            if re.search(pattern, answer, re.IGNORECASE):
                print(f"[LLM] WARNING: LLM tried to dodge when context exists, forcing re-answer is not possible, returning escalation")
                return ESCALATION_MSG

    return answer
