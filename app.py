"""
app.py — Flask backend for the AI Quiz Generator.

Routes
------
GET  /              → serve index.html
POST /upload        → accept a PDF, extract text, store in session
POST /generate-quiz → call brain.py and return structured quiz JSON
"""

import os
import json
from flask import Flask, request, jsonify, send_from_directory, session
from dotenv import load_dotenv
from brain import extract_text_from_bytes, generate_quiz

load_dotenv()

app = Flask(__name__, static_folder="static", template_folder=".")
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-change-in-prod")

# Max upload size: 16 MB
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = {"pdf"}


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# ---------------------------------------------------------------------------
# Serve the frontend
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


# ---------------------------------------------------------------------------
# POST /upload
# Accepts a PDF file, extracts its text, stores it in the session.
# ---------------------------------------------------------------------------

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF files are supported."}), 415

    try:
        pdf_bytes = file.read()
        text = extract_text_from_bytes(pdf_bytes)

        if not text.strip():
            return jsonify({"error": "Could not extract text from the PDF. "
                                     "Make sure it is not a scanned/image-only PDF."}), 422

        # Store extracted text in the server-side session
        session["extracted_text"] = text
        session["filename"] = file.filename

        return jsonify({
            "message": "File uploaded and text extracted successfully.",
            "filename": file.filename,
            "char_count": len(text),
        }), 200

    except Exception as e:
        return jsonify({"error": f"Failed to process PDF: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# POST /generate-quiz
# Reads text from session, calls brain.generate_quiz, returns JSON quiz.
# ---------------------------------------------------------------------------

@app.route("/generate-quiz", methods=["POST"])
def generate_quiz_route():
    text = session.get("extracted_text")

    if not text:
        return jsonify({"error": "No document uploaded. Please upload a PDF first."}), 400

    # Optional: caller can request a specific number of questions (default 5)
    body = request.get_json(silent=True) or {}
    try:
        num_questions = max(1, min(int(body.get("num_questions", 5)), 10))
    except (TypeError, ValueError):
        num_questions = 5

    try:
        quiz_data = generate_quiz(text, num_questions=num_questions)
        return jsonify(quiz_data), 200

    except Exception as e:
        return jsonify({"error": f"Quiz generation failed: {str(e)}"}), 500


# ---------------------------------------------------------------------------
# Health-check (optional, useful for testing)
# ---------------------------------------------------------------------------

@app.route("/health")
def health():
    return jsonify({"status": "ok"}), 200


if __name__ == "__main__":
    app.run(debug=True, port=5000)
