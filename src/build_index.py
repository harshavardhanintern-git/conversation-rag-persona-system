import argparse
import os
import json
import joblib
from sklearn.feature_extraction.text import HashingVectorizer

from parser import parse_csv
from topic_engine import build_topic_checkpoints
from checkpoints import build_100_message_checkpoints
from persona import extract_persona

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default="data/conversations.csv")
    parser.add_argument("--out", default="artifacts")
    parser.add_argument("--topic-threshold", type=float, default=0.45)
    parser.add_argument("--max-messages", type=int, default=0, help="Optional debug limit; 0 means all messages")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("Parsing CSV...")
    messages = parse_csv(args.csv)
    if args.max_messages and args.max_messages > 0:
        messages = messages[:args.max_messages]
    print(f"Parsed {len(messages)} messages")

    print("Building topic checkpoints with dynamic semantic segmentation...")
    topics = build_topic_checkpoints(messages, threshold=args.topic_threshold)

    print("Building 100-message checkpoints...")
    checkpoints = build_100_message_checkpoints(messages, size=100)

    print("Extracting persona...")
    persona = extract_persona(messages, target_speaker="User 1")

    print("Building lightweight retrieval vectors...")
    vectorizer = HashingVectorizer(
        n_features=2**16,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
        stop_words="english"
    )
    message_texts = [m["text"] for m in messages]
    topic_texts = [f"{' '.join(t['keywords'])} {t['summary']}" for t in topics]
    checkpoint_texts = [f"{' '.join(c['keywords'])} {c['summary']}" for c in checkpoints]

    message_matrix = vectorizer.transform(message_texts)
    topic_matrix = vectorizer.transform(topic_texts)
    checkpoint_matrix = vectorizer.transform(checkpoint_texts)

    with open(os.path.join(args.out, "messages.json"), "w", encoding="utf-8") as f:
        json.dump(messages, f, ensure_ascii=False)
    with open(os.path.join(args.out, "topics.json"), "w", encoding="utf-8") as f:
        json.dump(topics, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "checkpoints.json"), "w", encoding="utf-8") as f:
        json.dump(checkpoints, f, ensure_ascii=False, indent=2)
    with open(os.path.join(args.out, "persona.json"), "w", encoding="utf-8") as f:
        json.dump(persona, f, ensure_ascii=False, indent=2)

    joblib.dump(vectorizer, os.path.join(args.out, "vectorizer.joblib"))
    joblib.dump(message_matrix, os.path.join(args.out, "message_matrix.joblib"))
    joblib.dump(topic_matrix, os.path.join(args.out, "topic_matrix.joblib"))
    joblib.dump(checkpoint_matrix, os.path.join(args.out, "checkpoint_matrix.joblib"))

    print("Done.")
    print(f"Messages: {len(messages)}")
    print(f"Topic checkpoints: {len(topics)}")
    print(f"100-message checkpoints: {len(checkpoints)}")
    print(f"Persona extracted to {os.path.join(args.out, 'persona.json')}")

if __name__ == "__main__":
    main()
