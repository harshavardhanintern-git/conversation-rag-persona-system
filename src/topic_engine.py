from collections import defaultdict
from typing import Any, Dict, List
import math
import re

from sklearn.feature_extraction.text import HashingVectorizer

from summarizer import extractive_summary, keywords_from_text


# Stop words for token filtering
STOP = set(
    "a an the and or but if is are was were be been being to of in on for with as by from at it this that i you he she they we my your his her their our me him them us do does did have has had can could would should will just really very so not no yes yeah hey hi hello thanks thank okay ok"
    .split()
)


def tokens(text: str) -> set:
    """Return a set of meaningful tokens from *text* excluding stop words."""
    return set(w for w in re.findall(r"[a-z][a-z']{2,}", text.lower()) if w not in STOP)


def jaccard(a: set, b: set) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    return len(a & b) / max(len(a | b), 1)


def _build_contextual_vectors(messages: List[Dict[str, Any]], window: int = 5):
    """Build one semantic vector per message using nearby chat context.

    Single chat turns are often too short for stable similarity scoring, so each
    message vector includes nearby turns from the same conversation when the CSV
    parser provides conversation ids. The segmentation loop still compares each
    message vector against the active topic centroid before appending it.
    """
    docs = [""] * len(messages)
    grouped: Dict[Any, List[int]] = defaultdict(list)
    missing_conversation_ids = False

    for idx, msg in enumerate(messages):
        conversation_id = msg.get("conversation_id")
        if conversation_id is None:
            missing_conversation_ids = True
            break
        grouped[conversation_id].append(idx)

    if missing_conversation_ids:
        for idx in range(len(messages)):
            start = max(0, idx - window)
            end = min(len(messages), idx + window + 1)
            docs[idx] = " ".join(messages[j].get("text", "") for j in range(start, end))
    else:
        for indices in grouped.values():
            for local_idx, msg_idx in enumerate(indices):
                start = max(0, local_idx - window)
                end = min(len(indices), local_idx + window + 1)
                docs[msg_idx] = " ".join(
                    messages[j].get("text", "") for j in indices[start:end]
                )

    vectorizer = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words="english",
    )
    return vectorizer.transform(docs)


def _copy_vector(vector):
    if hasattr(vector, "multiply"):
        return vector.copy()

    import numpy as np

    return np.asarray(vector, dtype=float).ravel().copy()


def _vector_at(vectors, index: int):
    vector = vectors[index]
    if hasattr(vector, "multiply"):
        return vector

    import numpy as np

    return np.asarray(vector, dtype=float).ravel()


def _vector_norm(vector) -> float:
    if hasattr(vector, "multiply"):
        return math.sqrt(float(vector.multiply(vector).sum()))

    try:
        import numpy as np

        return float(np.linalg.norm(vector))
    except Exception:
        return 0.0


def _cosine_similarity(left, right) -> float:
    left_norm = _vector_norm(left)
    right_norm = _vector_norm(right)
    if left_norm == 0.0 or right_norm == 0.0:
        return 1.0

    if hasattr(left, "multiply"):
        dot = float(left.multiply(right).sum())
    else:
        import numpy as np

        dot = float(np.dot(left, right))
    return dot / (left_norm * right_norm)


def build_topic_checkpoints(
    messages: List[Dict[str, Any]],
    vectors=None,
    threshold: float = 0.45,
    min_topic_size: int = 8,
    merge_threshold: float = 0.65,
    min_final_topic_size: int = 5,
) -> List[Dict[str, Any]]:
    """Dynamic chronological topic segmentation.

    * A new topic is started when the semantic similarity of the incoming message
      falls below *threshold* and the current topic already contains at least
      *min_topic_size* messages.
    * Each message is compared with the rolling topic centroid before it is
      appended to the active topic.
    * Adjacent topics with semantic similarity above *merge_threshold* are merged.
    * Topics with fewer than *min_final_topic_size* messages are merged into the
      closest adjacent topic when possible, otherwise removed.
    """
    if not messages:
        return []

    if vectors is None:
        vectors = _build_contextual_vectors(messages)

    segments: List[Dict[str, Any]] = []

    def new_segment(index: int) -> Dict[str, Any]:
        return {
            "indices": [index],
            "vector_sum": _copy_vector(_vector_at(vectors, index)),
            "tokens": set(tokens(messages[index].get("text", ""))),
        }

    def append_to_segment(segment: Dict[str, Any], index: int) -> None:
        segment["indices"].append(index)
        segment["vector_sum"] = segment["vector_sum"] + _vector_at(vectors, index)
        segment["tokens"] |= tokens(messages[index].get("text", ""))

    def merge_segments(left: Dict[str, Any], right: Dict[str, Any]) -> Dict[str, Any]:
        left["indices"].extend(right["indices"])
        left["vector_sum"] = left["vector_sum"] + right["vector_sum"]
        left["tokens"] |= right["tokens"]
        return left

    def segment_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
        sim = _cosine_similarity(left["vector_sum"], right["vector_sum"])
        if sim == 1.0 and (not left["tokens"] or not right["tokens"]):
            return jaccard(left["tokens"], right["tokens"])
        return sim

    def merge_adjacent_similar(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        optimized: List[Dict[str, Any]] = []
        for segment in raw_segments:
            if optimized and segment_similarity(optimized[-1], segment) > merge_threshold:
                merge_segments(optimized[-1], segment)
            else:
                optimized.append(segment)
        return optimized

    def merge_or_remove_small(raw_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        idx = 0
        while idx < len(raw_segments):
            if len(raw_segments[idx]["indices"]) >= min_final_topic_size:
                idx += 1
                continue

            if len(raw_segments) == 1:
                raw_segments.pop(idx)
                continue

            if idx == 0:
                merged = merge_segments(raw_segments[idx], raw_segments[idx + 1])
                raw_segments[idx : idx + 2] = [merged]
                continue

            if idx == len(raw_segments) - 1:
                merge_segments(raw_segments[idx - 1], raw_segments[idx])
                raw_segments.pop(idx)
                idx = max(idx - 1, 0)
                continue

            previous_similarity = segment_similarity(raw_segments[idx - 1], raw_segments[idx])
            next_similarity = segment_similarity(raw_segments[idx], raw_segments[idx + 1])
            if previous_similarity >= next_similarity:
                merge_segments(raw_segments[idx - 1], raw_segments[idx])
                raw_segments.pop(idx)
                idx = max(idx - 1, 0)
            else:
                merged = merge_segments(raw_segments[idx], raw_segments[idx + 1])
                raw_segments[idx : idx + 2] = [merged]
        return raw_segments

    def close_topic(indices: List[int], topic_id: int) -> Dict[str, Any]:
        segment = [messages[i] for i in indices]
        return {
            "topic_id": topic_id,
            "start_msg_id": segment[0]["msg_id"],
            "end_msg_id": segment[-1]["msg_id"],
            "message_count": len(segment),
            "keywords": keywords_from_text([m["text"] for m in segment]),
            "summary": extractive_summary(segment, max_sentences=3),
            "sample_messages": [
                {"msg_id": m["msg_id"], "speaker": m["speaker"], "text": m["text"]}
                for m in segment[:3]
            ],
        }

    current_segment = new_segment(0)

    for idx in range(1, len(messages)):
        incoming_tokens = tokens(messages[idx].get("text", ""))
        sim = _cosine_similarity(_vector_at(vectors, idx), current_segment["vector_sum"])
        if sim == 1.0 and (not incoming_tokens or not current_segment["tokens"]):
            sim = jaccard(incoming_tokens, current_segment["tokens"])

        should_split = sim < threshold and len(current_segment["indices"]) >= min_topic_size
        if should_split:
            segments.append(current_segment)
            current_segment = new_segment(idx)
        else:
            append_to_segment(current_segment, idx)

    segments.append(current_segment)
    segments = merge_adjacent_similar(segments)
    segments = merge_or_remove_small(segments)
    segments = merge_adjacent_similar(segments)

    return [
        close_topic(segment["indices"], topic_id)
        for topic_id, segment in enumerate(segments, start=1)
    ]
