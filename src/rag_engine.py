from typing import Dict, Any
import numpy as np

def top_k(query_vec, matrix, items, k=3, min_score=0.30):
    sims = (matrix @ query_vec.T).toarray().ravel()
    if sims.size == 0:
        return []
    order = np.argsort(sims)[::-1][:k]
    results = []
    for i in order:
        score = float(sims[i])
        if score >= min_score:
            item = items[i]
            # HARD FILTERING for messages
            if "text" in item and "topic_id" not in item:
                text_lower = item["text"].lower()
                if len(item["text"]) < 25:
                    continue
                if "my name is" in text_lower or "hi " in text_lower or "hello" in text_lower:
                    continue
                # Skip if it's just a question without context
                if item["text"].endswith("?") and len(item["text"].split()) < 6:
                    continue
            results.append({**item, "score": score})
    return results

def answer_query(query: str, vectorizer, matrices: Dict[str, Any], artifacts: Dict[str, Any], persona: Dict[str, Any]) -> Dict[str, Any]:
    qv = vectorizer.transform([query])
    
    topic_hits = top_k(qv, matrices["topic_matrix"], artifacts["topics"], k=3, min_score=0.30)
    msg_hits = top_k(qv, matrices["message_matrix"], artifacts["messages"], k=5, min_score=0.30)
    
    q_lower = query.lower()
    persona_keywords = ["person", "persona", "habit", "habits", "talk", "communicat", "style", "traits", "who is", "character", "user", "about the user"]
    persona_mode = any(x in q_lower for x in persona_keywords)

    if persona_mode:
        answer = "Based on conversation evidence, here is the analytical profile of the user:\n\n"
        
        # Personality Traits
        traits = persona.get("personality_traits", [])
        if traits:
            answer += "#### 🧠 Ambitious & Goal-Oriented Traits\n"
            for t in traits[:3]:
                answer += f"- The user demonstrates **{t['label']}** through specific actions like: *\"{t['evidence']}\"* (Msg ID: {t['evidence_msg_id']})\n"
        
        # Habits
        habits = persona.get("habits", [])
        if habits:
            answer += "\n#### 🕒 Identified Life Events & Habits\n"
            for h in habits[:3]:
                answer += f"- Evidence indicates the user **{h['label']}**. Supporting data: *\"{h['evidence']}\"*\n"
                
        # Facts
        facts = persona.get("personal_facts", [])
        if facts:
            answer += "\n#### 📋 Contextual Personal Facts\n"
            for f in facts[:3]:
                answer += f"- **{f['label']}**: Validated by conversation entry: *\"{f['evidence']}\"*\n"
    else:
        # Context-based answer
        evidence = []
        for h in topic_hits:
            evidence.append(f"**Thematic Context (Topic {h['topic_id']})**: {h['summary']}")
        for h in msg_hits:
            evidence.append(f"**Direct Evidence (Msg {h['msg_id']})**: \"{h['text']}\"")
            
        if evidence:
            answer = "Based on conversation evidence retrieved from the history:\n\n" + "\n\n".join(evidence)
        else:
            answer = "I couldn't find enough high-signal conversation evidence to answer that specifically (no relevant matches above 0.30 similarity)."

    return {
        "answer": answer,
        "retrieved_topics": topic_hits,
        "retrieved_messages": msg_hits,
    }


