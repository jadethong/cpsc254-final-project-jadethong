"""
app.py — StudyScribe Flask backend.

Routes
------
GET  /                → serve index.html
POST /upload          → accept one or more PDFs, index into ChromaDB RAG
POST /generate-quiz   → RAG retrieval + OpenAI function calling → JSON quiz
POST /submit-answer   → record a user's answer, update concept stats in session
GET  /progress        → return session concept stats + weak-concept suggestions
DELETE /reset         → clear the RAG index and session state
GET  /health          → simple health check
"""

import os
from flask import Flask, request, jsonify, send_from_directory, session
from dotenv import load_dotenv

from brain import (
    index_documents,
    collection_size,
    generate_quiz,
    update_concept_stats,
    weak_concepts,
)

load_dotenv()

app = Flask(__name__, template_folder=".")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "studyscribe-dev-secret")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB total


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _allowed(filename: str) -> bool:
    return filename.lower().endswith(".pdf")


def _init_session():
    """Ensure session has the required keys."""
    if "concept_stats" not in session:
        session["concept_stats"] = {}
    if "indexed_files" not in session:
        session["indexed_files"] = []


# ---------------------------------------------------------------------------
# GET /
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ---------------------------------------------------------------------------
# POST /upload
# Accepts one or more PDF files, indexes them into the local ChromaDB RAG.
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    _init_session()

    files = request.files.getlist("files")  # multiple files via <input multiple>
    if not files or all(f.filename == "" for f in files):
        return jsonify({"error": "No files received. Please select at least one PDF."}), 400

    pdf_list: list[tuple[str, bytes]] = []
    rejected: list[str] = []

    for f in files:
        if not _allowed(f.filename):
            rejected.append(f.filename)
            continue
        pdf_list.append((f.filename, f.read()))

    if not pdf_list:
        return jsonify({
            "error": "No valid PDF files found.",
            "rejected": rejected,
        }), 415

    try:
        num_chunks = index_documents(pdf_list)
    except Exception as e:
        return jsonify({"error": f"Indexing failed: {str(e)}"}), 500

    # Reset concept stats whenever new docs are loaded
    session["concept_stats"] = {}
    session["indexed_files"] = [name for name, _ in pdf_list]
    session.modified = True

    return jsonify({
        "message": f"Indexed {len(pdf_list)} file(s) into {num_chunks} chunks.",
        "files": session["indexed_files"],
        "chunks": num_chunks,
        "rejected": rejected,
    }), 200


# ---------------------------------------------------------------------------
# POST /generate-quiz
# Body (JSON): { "topic": "TCP vs UDP", "num_questions": 5 }
# ---------------------------------------------------------------------------

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz_route():
    _init_session()

    if collection_size() == 0:
        return jsonify({
            "error": "No documents indexed. Please upload your lecture PDFs first."
        }), 400

    body = request.get_json(silent=True) or {}
    topic = (body.get("topic") or "").strip()
    if not topic:
        return jsonify({"error": "Please provide a 'topic' in the request body."}), 400

    try:
        num_q = max(1, min(int(body.get("num_questions", 5)), 10))
    except (TypeError, ValueError):
        num_q = 5

    try:
        quiz_data = generate_quiz(topic=topic, num_questions=num_q)
    except Exception as e:
        return jsonify({"error": f"Quiz generation failed: {str(e)}"}), 500

    return jsonify(quiz_data), 200


# ---------------------------------------------------------------------------
# POST /submit-answer
# Body (JSON): { "concept": "TCP handshake", "correct": true }
# Records the result and returns updated weak-concept list.
# ---------------------------------------------------------------------------

@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    _init_session()

    body = request.get_json(silent=True) or {}
    concept = (body.get("concept") or "").strip()
    correct = bool(body.get("correct", False))

    if not concept:
        return jsonify({"error": "Missing 'concept' field."}), 400

    stats = session.get("concept_stats", {})
    stats = update_concept_stats(stats, concept, correct)
    session["concept_stats"] = stats
    session.modified = True

    return jsonify({
        "concept": concept,
        "correct": correct,
        "stats": stats.get(concept),
        "weak_concepts": weak_concepts(stats),
    }), 200


# ---------------------------------------------------------------------------
# GET /progress
# Returns full concept stats and weak-concept suggestions for the session.
# ---------------------------------------------------------------------------

@app.route("/progress", methods=["GET"])
def progress():
    _init_session()

    stats = session.get("concept_stats", {})
    total_answered = sum(v["total"] for v in stats.values())
    total_correct  = sum(v["correct"] for v in stats.values())

    return jsonify({
        "indexed_files": session.get("indexed_files", []),
        "total_answered": total_answered,
        "total_correct": total_correct,
        "overall_accuracy": round(total_correct / total_answered, 2) if total_answered else None,
        "concept_stats": stats,
        "weak_concepts": weak_concepts(stats),
    }), 200


# ---------------------------------------------------------------------------
# DELETE /reset
# Clears the RAG index and session state (useful for starting fresh).
# ---------------------------------------------------------------------------

@app.route("/reset", methods=["DELETE"])
def reset():
    try:
        # Re-index with an empty list just to wipe the collection
        index_documents([])
    except Exception:
        pass

    session.clear()
    return jsonify({"message": "Session and index cleared."}), 200


# ---------------------------------------------------------------------------
# GET /health
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({
        "status": "ok",
        "indexed_chunks": collection_size(),
    }), 200


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True, port=5000)
