from typing import List, Dict, Any
from summarizer import extractive_summary, keywords_from_text

def build_100_message_checkpoints(messages: List[Dict[str, Any]], size: int = 100) -> List[Dict[str, Any]]:
    checkpoints = []
    checkpoint_id = 1
    for start in range(0, len(messages), size):
        segment = messages[start:start+size]
        if not segment:
            continue
        checkpoints.append({
            "checkpoint_id": checkpoint_id,
            "start_msg_id": segment[0]["msg_id"],
            "end_msg_id": segment[-1]["msg_id"],
            "message_count": len(segment),
            "keywords": keywords_from_text([m["text"] for m in segment]),
            "summary": extractive_summary(segment, max_sentences=4)
        })
        checkpoint_id += 1
    return checkpoints
