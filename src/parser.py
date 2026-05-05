import re
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import pandas as pd

SPEAKER_RE = re.compile(r"^(User\s*\d+)\s*:\s*(.*)$", re.IGNORECASE)

@dataclass
class Message:
    msg_id: int
    conversation_id: int
    local_msg_id: int
    speaker: str
    text: str

def parse_csv(csv_path: str) -> List[Dict[str, Any]]:
    df = pd.read_csv(csv_path, header=None)
    messages: List[Message] = []
    global_id = 1

    for conv_id, raw in enumerate(df.iloc[:, 0].fillna("").astype(str).tolist(), start=1):
        local_id = 1
        current_speaker = None
        current_text_parts = []

        def flush():
            nonlocal global_id, local_id, current_speaker, current_text_parts
            text = " ".join([p.strip() for p in current_text_parts if p.strip()]).strip()
            if current_speaker and text:
                messages.append(Message(global_id, conv_id, local_id, current_speaker, text))
                global_id += 1
                local_id += 1
            current_text_parts = []

        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            m = SPEAKER_RE.match(line)
            if m:
                flush()
                current_speaker = m.group(1).title().replace("  ", " ")
                current_text_parts = [m.group(2).strip()]
            else:
                current_text_parts.append(line)
        flush()

    return [asdict(m) for m in messages]
