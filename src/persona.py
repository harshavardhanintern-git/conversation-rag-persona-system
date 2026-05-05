from typing import List, Dict, Any
import re

def add_item(bucket, label, evidence_msg, confidence="High"):
    bucket.append({
        "label": label,
        "evidence_msg_id": evidence_msg["msg_id"],
        "evidence": evidence_msg["text"],
        "confidence": confidence
    })

GREETINGS = {"hi", "hello", "hey", "sup", "yo", "morning", "evening", "thanks", "thank", "ok", "okay", "yeah", "yes", "no"}

def is_meaningful(text: str) -> bool:
    if len(text) < 25:
        return False
    tokens = re.findall(r"\w+", text.lower())
    if not tokens:
        return False
    # Filter greetings and generic filler phrases
    generic = {"hi", "hello", "hey", "sup", "yo", "morning", "evening", "thanks", "thank", "ok", "okay", "yeah", "yes", "no", "how are you", "what's up", "good", "fine"}
    clean_text = " ".join(tokens)
    if any(g == clean_text for g in generic):
        return False
    if all(t in GREETINGS for t in tokens):
        return False
    return True

def extract_persona(messages: List[Dict[str, Any]], target_speaker: str = "User 1") -> Dict[str, Any]:
    user_msgs = [m for m in messages if m["speaker"].lower() == target_speaker.lower()]
    meaningful_msgs = [m for m in user_msgs if is_meaningful(m["text"])]
    
    persona = {
        "target_speaker": target_speaker,
        "habits": [],
        "personal_facts": [],
        "personality_traits": [],
        "communication_style": [],
        "stats": {
            "total_messages": len(user_msgs),
            "meaningful_messages": len(meaningful_msgs),
            "avg_message_length": round(sum(len(m["text"]) for m in user_msgs) / max(len(user_msgs), 1), 2),
        }
    }

    if not meaningful_msgs:
        return persona

    # Communication Style - Only if very strong pattern
    question_msgs = [m for m in meaningful_msgs if "?" in m["text"]]
    if len(question_msgs) > len(meaningful_msgs) * 0.3:
        add_item(persona["communication_style"], "Analytical communicator focused on information gathering", question_msgs[0], "High")

    # Habits & Facts - Strong goal-oriented and life event filters
    patterns = [
        ("culinary_goals", r"\bcook\b|\bculinary\b|\bchef\b|\brestaurant\b", "Committed to professional culinary development", "habits"),
        ("education_path", r"\blearn\b|\bstudy\b|\bcollege\b|\bdegree\b|\buniversity\b", "Actively pursuing higher education or specialized training", "habits"),
        ("relocation", r"\bmoving to\b|\bmove to\b|\bmoved to\b", "Managing a significant life transition (relocation)", "personal_facts"),
        ("career_trajectory", r"\bpursuing\b|\bdreams\b|\bcareer\b|\bjob\b|\binternship\b", "Focused on long-term professional aspirations and career growth", "personal_facts"),
        ("family_relations", r"\bfamily\b|\bmother\b|\bfather\b|\bsister\b|\bbrother\b|\bparents\b", "Maintains strong focus on familial relationships and dynamics", "personal_facts"),
    ]

    seen = set()
    for m in meaningful_msgs:
        text = m["text"].lower()
        for key, regex, label, bucket in patterns:
            if key not in seen and re.search(regex, text):
                # Verify it's not a short identity message
                if "my name is" in text:
                    continue
                add_item(persona[bucket], label, m, "High")
                seen.add(key)

    # Personality Traits - REQUIRE MINIMUM 2 EVIDENCE POINTS
    trait_patterns = [
        ("Ambitious and Goal-Oriented", ["pursuing", "dream", "career", "goal", "striving"]),
        ("Resilient and Adaptive", ["moving", "new city", "change", "transition", "handle"]),
        ("Emotionally Articulate", ["worried", "nervous", "excited", "feel", "scared"]),
    ]
    for label, words in trait_patterns:
        matches = [m for m in meaningful_msgs if any(w in m["text"].lower() for w in words)]
        if len(matches) >= 2:  # Strict requirement for at least 2 messages
            add_item(persona["personality_traits"], label, matches[0], "High")
            
    return persona


