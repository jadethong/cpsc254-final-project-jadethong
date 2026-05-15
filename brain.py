"""
brain.py — StudyScribe AI core.

Pipeline
--------
1. PDF text is chunked and stored in a local ChromaDB collection (RAG index).
2. On quiz generation, the user's topic is embedded and the top-k relevant
   chunks are retrieved.
3. The retrieved context is passed to gpt-4o-mini with OpenAI Function
   Calling (tool_choice forced to `generate_quiz_json`) to guarantee a
   clean JSON quiz.
4. If the topic is not grounded in the uploaded documents, the model is
   instructed to return a refusal instead of using base knowledge.
5. Session concept tracking: answer results are recorded per concept so
   the app can surface weak areas.

Everything runs locally — no external vector DB, no Pinecone, no Supabase.
"""

from __future__ import annotations

import io
import json
import os
import uuid
import textwrap
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Clients
# ---------------------------------------------------------------------------

_openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Persistent local ChromaDB stored in ./chroma_db/
_chroma_client = chromadb.PersistentClient(path="./chroma_db")

_embed_fn = OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"),
    model_name="text-embedding-3-small",
)

# One collection per run; reset when new documents are uploaded
_COLLECTION_NAME = "studyscribe_docs"


def _get_collection() -> chromadb.Collection:
    return _chroma_client.get_or_create_collection(
        name=_COLLECTION_NAME,
        embedding_function=_embed_fn,
    )


# ---------------------------------------------------------------------------
# OpenAI Function Calling tool definition
# ---------------------------------------------------------------------------

QUIZ_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_quiz_json",
        "description": (
            "Generate a multiple-choice quiz strictly from the provided lecture context. "
            "If the requested topic is not covered by the context, set 'out_of_scope' to true "
            "and populate 'refusal_message' instead of questions."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "out_of_scope": {
                    "type": "boolean",
                    "description": (
                        "Set to true when the requested topic is NOT covered by the "
                        "retrieved course materials."
                    ),
                },
                "refusal_message": {
                    "type": "string",
                    "description": (
                        "Polite explanation returned when out_of_scope is true. "
                        "Must mention which topic was requested and remind the user "
                        "that only uploaded materials are used."
                    ),
                },
                "topic": {
                    "type": "string",
                    "description": "Short label for the quiz topic, derived from the context.",
                },
                "questions": {
                    "type": "array",
                    "description": "List of quiz questions (empty when out_of_scope is true).",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {"type": "string"},
                            "choices": {
                                "type": "array",
                                "items": {"type": "string"},
                                "minItems": 4,
                                "maxItems": 4,
                                "description": "Exactly 4 choices labelled 'A. ', 'B. ', 'C. ', 'D. '.",
                            },
                            "answer": {
                                "type": "string",
                                "description": "Must match one of the choices exactly.",
                            },
                            "explanation": {
                                "type": "string",
                                "description": "Brief explanation grounded in the retrieved context.",
                            },
                            "concept": {
                                "type": "string",
                                "description": (
                                    "Short concept tag for this question (e.g. 'TCP handshake', "
                                    "'RSA key pair'). Used for concept tracking."
                                ),
                            },
                        },
                        "required": ["question", "choices", "answer", "explanation", "concept"],
                    },
                },
            },
            "required": ["out_of_scope", "questions"],
        },
    },
}


# ---------------------------------------------------------------------------
# PDF → chunks
# ---------------------------------------------------------------------------

def _extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages).strip()


def _chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks, start = [], 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return [c.strip() for c in chunks if c.strip()]


# ---------------------------------------------------------------------------
# Public: index documents
# ---------------------------------------------------------------------------

def index_documents(pdf_bytes_list: list[tuple[str, bytes]]) -> int:
    """
    Clear the existing collection and index all provided PDFs.

    Parameters
    ----------
    pdf_bytes_list : list of (filename, bytes)

    Returns
    -------
    Total number of chunks indexed.
    """
    # Delete and recreate the collection for a clean slate
    try:
        _chroma_client.delete_collection(_COLLECTION_NAME)
    except Exception:
        pass
    collection = _get_collection()

    all_chunks, all_ids, all_metas = [], [], []
    for filename, pdf_bytes in pdf_bytes_list:
        text = _extract_text(pdf_bytes)
        chunks = _chunk_text(text)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(f"{filename}_chunk_{i}_{uuid.uuid4().hex[:6]}")
            all_metas.append({"source": filename, "chunk_index": i})

    if all_chunks:
        # ChromaDB has a max batch size of 5461
        batch = 500
        for i in range(0, len(all_chunks), batch):
            collection.add(
                documents=all_chunks[i : i + batch],
                ids=all_ids[i : i + batch],
                metadatas=all_metas[i : i + batch],
            )

    return len(all_chunks)


def collection_size() -> int:
    """Return number of chunks currently indexed."""
    try:
        return _get_collection().count()
    except Exception:
        return 0


# ---------------------------------------------------------------------------
# Public: RAG-based quiz generation
# ---------------------------------------------------------------------------

def generate_quiz(topic: str, num_questions: int = 5) -> dict:
    """
    Retrieve relevant chunks for `topic` and generate a quiz via function calling.

    Returns
    -------
    dict with keys:
      - out_of_scope  (bool)
      - refusal_message (str, only when out_of_scope)
      - topic  (str)
      - questions  (list)
    """
    if collection_size() == 0:
        return {
            "out_of_scope": True,
            "refusal_message": "No documents have been uploaded yet. Please upload your lecture PDFs first.",
            "topic": topic,
            "questions": [],
        }

    # --- Retrieve top-k relevant chunks ---
    collection = _get_collection()
    k = min(6, collection.count())
    results = collection.query(query_texts=[topic], n_results=k)
    retrieved_chunks = results["documents"][0] if results["documents"] else []
    sources = list({m["source"] for m in results["metadatas"][0]}) if results["metadatas"] else []

    context_block = "\n\n---\n\n".join(retrieved_chunks)

    system_prompt = textwrap.dedent(f"""
        You are StudyScribe, a strict retrieval-based tutor. Your ONLY knowledge source is the
        retrieved course material provided. You must ignore your base training knowledge
        regarding facts, historical titles, or grammar rules if they are not in the text.

        Rules:
        1. If the requested topic is not explicitly covered by the retrieved context,
           set out_of_scope=true and write a polite refusal_message.
        2. Every answer and explanation must be directly traceable to a specific sentence 
           in the retrieved context. No outside facts.
        3. Each choice must be labelled exactly 'A. ', 'B. ', 'C. ', 'D. '.
        4. The answer field must match one of the choices exactly.
        5. Use the EXACT terminology found in the context.
        6. CATEGORIZATION RIGOR: If the text groups items into specific categories (e.g., 
           'Time Signals' vs 'Importance Signals'), do not create questions that treat 
           them as interchangeable or 'All of the above' unless the text explicitly 
           groups them that way.
        7. NO LOGIC LEAKAGE: Do not assume a relationship between terms unless the 
           text explicitly states it.

        Retrieved course material (sources: {', '.join(sources) or 'uploaded docs'}):
        ===
        {context_block[:10_000]}
        ===
    """).strip()

    user_prompt = (
        f"Create {num_questions} multiple-choice questions about: {topic}"
    )

    response = _openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[QUIZ_TOOL],
        tool_choice={"type": "function", "function": {"name": "generate_quiz_json"}},
        temperature=0.4,
    )

    tool_call = response.choices[0].message.tool_calls[0]
    quiz_data = json.loads(tool_call.function.arguments)

    # Attach source list for transparency
    quiz_data.setdefault("sources", sources)
    quiz_data.setdefault("topic", topic)
    quiz_data.setdefault("questions", [])
    return quiz_data


# ---------------------------------------------------------------------------
# Session concept tracking
# ---------------------------------------------------------------------------

def update_concept_stats(
    session_stats: dict,
    concept: str,
    correct: bool,
) -> dict:
    """
    Update in-memory concept stats dict.

    session_stats shape:
    {
        "concept_name": {"correct": int, "total": int},
        ...
    }
    Returns the updated dict.
    """
    if concept not in session_stats:
        session_stats[concept] = {"correct": 0, "total": 0}
    session_stats[concept]["total"] += 1
    if correct:
        session_stats[concept]["correct"] += 1
    return session_stats


def weak_concepts(session_stats: dict, threshold: float = 0.6) -> list[dict]:
    """
    Return concepts where accuracy < threshold, sorted worst-first.
    Only includes concepts with ≥ 1 attempt.
    """
    weak = []
    for concept, stats in session_stats.items():
        if stats["total"] == 0:
            continue
        acc = stats["correct"] / stats["total"]
        if acc < threshold:
            weak.append({
                "concept": concept,
                "accuracy": round(acc, 2),
                "correct": stats["correct"],
                "total": stats["total"],
            })
    return sorted(weak, key=lambda x: x["accuracy"])
