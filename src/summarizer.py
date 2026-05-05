from typing import List, Dict, Any
import re
from collections import Counter

STOPWORDS = set("""
a an the and or but if is are was were be been being to of in on for with as by from at it this that i you he she they we my your his her their our me him them us
do does did have has had can could would should will just really very so not no yes yeah hey hi hello thanks thank okay ok
""".split())

def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()

def keywords_from_text(texts: List[str], top_k: int = 8) -> List[str]:
    words = []
    for t in texts:
        words += re.findall(r"[A-Za-z][A-Za-z']{2,}", t.lower())
    counts = Counter(w for w in words if w not in STOPWORDS)
    return [w for w, _ in counts.most_common(top_k)]

def extractive_summary(messages: List[Dict[str, Any]], max_sentences: int = 3) -> str:
    if not messages:
        return ""
    texts = [clean_text(m["text"]) for m in messages if clean_text(m["text"])]
    keys = set(keywords_from_text(texts, top_k=12))
    scored = []
    for idx, text in enumerate(texts):
        words = set(re.findall(r"[A-Za-z][A-Za-z']{2,}", text.lower()))
        score = len(words & keys) + (1 if len(text) > 20 else 0)
        scored.append((score, idx, text))
    chosen = sorted(scored, reverse=True)[:max_sentences]
    chosen = sorted(chosen, key=lambda x: x[1])
    return " ".join(x[2] for x in chosen)
