"""
brain.py — AI logic for quiz generation using OpenAI Function Calling.
Extracts text from a PDF, then calls the OpenAI API with a tool definition
for `generate_quiz_json` to guarantee structured JSON output.
"""

import json
import os
from openai import OpenAI
from pypdf import PdfReader
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------------------------------------------------------------------------
# Tool definition (OpenAI Function Calling)
# ---------------------------------------------------------------------------

QUIZ_TOOL = {
    "type": "function",
    "function": {
        "name": "generate_quiz_json",
        "description": (
            "Generate a multiple-choice quiz from the provided lecture content. "
            "Return exactly the number of questions requested."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array",
                    "description": "List of quiz questions.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "question": {
                                "type": "string",
                                "description": "The quiz question text."
                            },
                            "choices": {
                                "type": "array",
                                "description": "Exactly 4 answer choices (A–D).",
                                "items": {"type": "string"},
                                "minItems": 4,
                                "maxItems": 4
                            },
                            "answer": {
                                "type": "string",
                                "description": "The correct answer, matching one of the choices exactly."
                            },
                            "explanation": {
                                "type": "string",
                                "description": "A brief explanation of why the answer is correct."
                            }
                        },
                        "required": ["question", "choices", "answer", "explanation"]
                    }
                }
            },
            "required": ["questions"]
        }
    }
}


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_text_from_pdf(pdf_path: str) -> str:
    """Return all text content from a PDF file."""
    reader = PdfReader(pdf_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


def extract_text_from_bytes(pdf_bytes: bytes) -> str:
    """Return all text content from in-memory PDF bytes."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages).strip()


# ---------------------------------------------------------------------------
# Core quiz generation
# ---------------------------------------------------------------------------

def generate_quiz(text: str, num_questions: int = 5) -> dict:
    """
    Call the OpenAI API with function calling to produce a structured quiz.

    Parameters
    ----------
    text          : The lecture/transcript text to quiz on.
    num_questions : How many questions to generate (default 5).

    Returns
    -------
    A dict with a "questions" key containing the quiz data, e.g.:
    {
        "questions": [
            {
                "question": "...",
                "choices": ["A. ...", "B. ...", "C. ...", "D. ..."],
                "answer": "A. ...",
                "explanation": "..."
            },
            ...
        ]
    }
    """
    # Truncate very long texts to stay within token limits (~12 000 chars ≈ 3 000 tokens)
    max_chars = 12_000
    if len(text) > max_chars:
        text = text[:max_chars] + "\n\n[Content truncated for length]"

    system_prompt = (
        "You are an expert educator. "
        "Given lecture content, you create clear, accurate, multiple-choice quiz questions "
        "that test conceptual understanding. "
        "Each question must have exactly 4 choices labelled 'A. ', 'B. ', 'C. ', 'D. '. "
        "The answer field must match one of the choices exactly."
    )

    user_prompt = (
        f"Create {num_questions} multiple-choice questions based on the following lecture content.\n\n"
        f"--- LECTURE CONTENT ---\n{text}\n--- END ---"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tools=[QUIZ_TOOL],
        tool_choice={"type": "function", "function": {"name": "generate_quiz_json"}},
        temperature=0.7,
    )

    # Extract the function call arguments
    tool_call = response.choices[0].message.tool_calls[0]
    quiz_data = json.loads(tool_call.function.arguments)
    return quiz_data


# ---------------------------------------------------------------------------
# Convenience wrapper for file paths
# ---------------------------------------------------------------------------

def generate_quiz_from_pdf(pdf_path: str, num_questions: int = 5) -> dict:
    """Extract text from a PDF file and generate a quiz."""
    text = extract_text_from_pdf(pdf_path)
    if not text:
        raise ValueError(f"Could not extract text from PDF: {pdf_path}")
    return generate_quiz(text, num_questions)
