# Conversation RAG + Persona Extraction System

A system that processes conversation data in chronological order and builds a Retrieval-Augmented Generation (RAG) pipeline with dynamic topic checkpoints and evidence-based persona extraction.

---

## 🚀 Key Features

### 1. Chronological Topic Checkpoints

- Conversations are processed **message by message in chronological order**
- Topic changes are detected using **embedding similarity**
- A new topic is created when:
  - similarity < **0.45**
  - current topic size ≥ **8 messages**
- Adjacent topics are merged when similarity > **0.65**
- Topics with very few messages are merged or removed

👉 This ensures topics represent meaningful conversation segments instead of fixed chunks.

---

### 2. 100-Message Checkpoints

- For every **100 messages**, a summary is generated
- These checkpoints are **independent of topics**
- Used as an additional retrieval layer for long conversations

---

### 3. Retrieval Pipeline

When a user asks a question:

- Retrieve relevant **topic summaries**
- Retrieve relevant **message chunks**
- Apply filtering:
  - similarity score > **0.30**
  - remove short or irrelevant messages
- Combine both sources to generate the final answer

👉 This ensures retrieval is relevant and not random.

---

### 4. Persona Extraction (Evidence-Based)

Persona is extracted using **rule-based filtering + conversation signals**.

#### Filtering Rules:
- Ignore messages < **25 characters**
- Remove greetings and filler text
- Only consider meaningful content

#### Trait Extraction:
- Each trait requires **minimum 2 supporting messages**
- Traits are derived from:
  - habits
  - personal facts
  - personality traits
  - communication style

#### Output:
- Stored in structured **JSON format**
- Each trait includes:
  - label
  - evidence (message text)
  - message ID
  - confidence level

👉 No assumptions — only evidence-driven insights.

---

### 5. Chatbot

The system answers:

- What kind of person is this user?
- What are their habits?
- How do they talk?

It uses:
- RAG pipeline (topics + messages)
- Persona data

👉 Answers are grounded in retrieved evidence.

---

## 🛠️ How to Run

```bash
pip install -r requirements.txt
python src/build_index.py --csv data/conversations.csv
streamlit run app.py
