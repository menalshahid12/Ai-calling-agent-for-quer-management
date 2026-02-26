"""
RAG knowledge base: loads IST data files, builds ChromaDB vector index
(when available), and provides hybrid (vector + keyword) search.
Falls back to keyword-only search if ChromaDB fails.
"""

import os, re, hashlib
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")

_chroma_collection = None
_documents: list[dict] = []
_chroma_available = False


def _chunk_text(text: str, title: str, max_chars: int = 800) -> list[dict]:
    paragraphs = re.split(r"\n{2,}", text.strip())
    chunks, buf = [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 > max_chars and buf:
            chunks.append({"text": buf.strip(), "title": title})
            buf = p + "\n\n"
        else:
            buf += p + "\n\n"
    if buf.strip():
        chunks.append({"text": buf.strip(), "title": title})
    return chunks


def _load_documents() -> list[dict]:
    docs = []
    for fname in sorted(os.listdir(DATA_DIR)):
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(fpath):
            continue
        if fname.endswith(".json"):
            continue
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            continue
        title = Path(fname).stem.replace("_", " ").title()
        docs.extend(_chunk_text(text, title))
    return docs


def _build_chroma():
    global _chroma_collection, _chroma_available
    try:
        import chromadb

        try:
            client = chromadb.PersistentClient(path=CHROMA_DIR)
        except Exception:
            client = chromadb.Client()

        try:
            client.delete_collection("ist_kb")
        except Exception:
            pass

        _chroma_collection = client.get_or_create_collection(
            name="ist_kb",
            metadata={"hnsw:space": "cosine"},
        )

        ids, texts, metas = [], [], []
        for i, doc in enumerate(_documents):
            doc_id = hashlib.md5(f"{i}_{doc['text'][:200]}".encode()).hexdigest()
            ids.append(doc_id)
            texts.append(doc["text"])
            metas.append({"title": doc["title"]})

        batch = 50
        for start in range(0, len(ids), batch):
            end = start + batch
            _chroma_collection.add(
                ids=ids[start:end],
                documents=texts[start:end],
                metadatas=metas[start:end],
            )
        _chroma_available = True
        print(f"[KB] ChromaDB built with {len(ids)} chunks")
    except Exception as e:
        print(f"[KB] ChromaDB unavailable ({e}), using keyword search only")
        _chroma_available = False


def init_kb():
    global _documents
    _documents = _load_documents()
    print(f"[KB] Loaded {len(_documents)} chunks from {DATA_DIR}")
    _build_chroma()


def _keyword_search(query: str, top_k: int = 8) -> list[dict]:
    query_lower = query.lower()
    tokens = set(re.findall(r"\w{3,}", query_lower))
    scored = []
    for doc in _documents:
        text_lower = doc["text"].lower()
        score = sum(2 for t in tokens if t in text_lower)
        for phrase in _extract_phrases(query_lower):
            if phrase in text_lower:
                score += 5
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: -x[0])
    return [s[1] for s in scored[:top_k]]


def _extract_phrases(text: str) -> list[str]:
    words = text.split()
    phrases = [text]
    if len(words) >= 2:
        for i in range(len(words) - 1):
            phrases.append(f"{words[i]} {words[i+1]}")
    return phrases


def search(query: str, top_k: int = 8) -> str:
    results = []

    if _chroma_available and _chroma_collection is not None:
        try:
            res = _chroma_collection.query(query_texts=[query], n_results=min(top_k, 6))
            if res and res.get("documents"):
                for doc_text, meta in zip(res["documents"][0], res["metadatas"][0]):
                    results.append({"text": doc_text, "title": meta.get("title", "")})
        except Exception as e:
            print(f"[KB] Chroma search error: {e}")

    kw_results = _keyword_search(query, top_k=6)
    seen_texts = {r["text"][:100] for r in results}
    for kw in kw_results:
        if kw["text"][:100] not in seen_texts:
            results.append(kw)
            seen_texts.add(kw["text"][:100])

    if not results:
        fallback_q = "admission programs fee merit IST eligibility department"
        results = _keyword_search(fallback_q, top_k=4)

    results = results[:top_k]
    context_parts = []
    for r in results:
        context_parts.append(f"[{r['title']}]\n{r['text']}")
    return "\n\n---\n\n".join(context_parts) if context_parts else "No relevant information found in the knowledge base."
