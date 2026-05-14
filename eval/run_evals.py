"""
eval/run_evals.py — Evaluation harness for the AI Quiz Generator.

Metric: Question_Relevance_Score
---------------------------------
For each test case, we call brain.generate_quiz() on the provided context.
We then check how many of the expected_keywords appear across all generated
question texts and answers.

  keyword_hits  = number of expected keywords found in the generated quiz
  keyword_total = total number of expected keywords

  Question_Relevance_Score (per case) = keyword_hits / keyword_total

  Overall Score = mean of all per-case scores (0.0 – 1.0)

Usage
-----
  cd <project-root>
  python eval/run_evals.py [--cases eval/test_cases.json] [--num-questions 5] [--verbose]
"""

import argparse
import json
import os
import sys
import time

# Make sure the project root is on the path so we can import brain.py
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from brain import generate_quiz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def quiz_text(questions: list[dict]) -> str:
    """Flatten all question + answer text into a single searchable string."""
    parts = []
    for q in questions:
        parts.append(q.get("question", ""))
        parts.extend(q.get("choices", []))
        parts.append(q.get("answer", ""))
        parts.append(q.get("explanation", ""))
    return " ".join(parts).lower()


def relevance_score(quiz_blob: str, expected_keywords: list[str]) -> tuple[float, list[str], list[str]]:
    """
    Returns (score, found_keywords, missing_keywords).
    score is in [0.0, 1.0].
    """
    found   = [kw for kw in expected_keywords if kw.lower() in quiz_blob]
    missing = [kw for kw in expected_keywords if kw.lower() not in quiz_blob]
    score   = len(found) / len(expected_keywords) if expected_keywords else 0.0
    return score, found, missing


def bar(score: float, width: int = 20) -> str:
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evals(cases_path: str, num_questions: int, verbose: bool) -> None:
    with open(cases_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    print("=" * 65)
    print("  AI Quiz Generator — Evaluation Report")
    print(f"  Cases: {len(test_cases)}  |  Questions per case: {num_questions}")
    print("=" * 65)

    scores      = []
    errors      = []
    total_start = time.time()

    for case in test_cases:
        case_id  = case["id"]
        topic    = case["topic"]
        context  = case["context"]
        keywords = case["expected_keywords"]

        print(f"\n[Case {case_id:02d}] {topic}")
        print(f"         Keywords to find: {keywords}")

        start = time.time()
        try:
            quiz_data = generate_quiz(context, num_questions=num_questions)
            elapsed   = time.time() - start

            blob  = quiz_text(quiz_data.get("questions", []))
            score, found, missing = relevance_score(blob, keywords)

            scores.append(score)

            status = "✅" if score >= 0.6 else ("⚠️ " if score >= 0.3 else "❌")
            print(f"         Question_Relevance_Score: {bar(score)}  {status}  ({elapsed:.1f}s)")

            if verbose:
                print(f"         Found   : {found}")
                print(f"         Missing : {missing}")
                print()
                for i, q in enumerate(quiz_data.get("questions", []), 1):
                    print(f"           Q{i}: {q['question']}")
                    for c in q.get("choices", []):
                        marker = "✓" if c == q["answer"] else " "
                        print(f"              [{marker}] {c}")
                    print(f"              Explanation: {q.get('explanation', '')}")
                    print()

        except Exception as e:
            elapsed = time.time() - start
            errors.append({"id": case_id, "topic": topic, "error": str(e)})
            print(f"         ❌ ERROR ({elapsed:.1f}s): {e}")
            scores.append(0.0)

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    overall       = sum(scores) / len(scores) if scores else 0.0

    print("\n" + "=" * 65)
    print("  SUMMARY")
    print("=" * 65)
    print(f"  Cases run     : {len(test_cases)}")
    print(f"  Errors        : {len(errors)}")
    print(f"  Total time    : {total_elapsed:.1f}s")
    print()
    print(f"  Overall Question_Relevance_Score: {bar(overall)}")
    print()

    # Per-case table
    print(f"  {'ID':<5} {'Topic':<40} {'Score':>6}")
    print(f"  {'-'*5} {'-'*40} {'-'*6}")
    for i, (case, score) in enumerate(zip(test_cases, scores)):
        flag = "❌" if i < len(errors) and errors[i]["id"] == case["id"] else ""
        print(f"  {case['id']:<5} {case['topic'][:40]:<40} {score:>5.0%}  {flag}")

    print("=" * 65)

    if errors:
        print("\nFailed cases:")
        for err in errors:
            print(f"  Case {err['id']} — {err['topic']}: {err['error']}")

    # Exit with non-zero if overall score is below threshold
    threshold = 0.5
    if overall < threshold:
        print(f"\n⚠️  Overall score {overall:.0%} is below threshold {threshold:.0%}.")
        sys.exit(1)
    else:
        print(f"\n✅ Evaluation passed (score {overall:.0%} ≥ threshold {threshold:.0%}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run quiz-generation evals.")
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases.json"),
        help="Path to test_cases.json (default: eval/test_cases.json)",
    )
    parser.add_argument(
        "--num-questions", type=int, default=3,
        help="Number of questions to generate per test case (default: 3)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print every generated question and answer",
    )
    args = parser.parse_args()

    run_evals(
        cases_path=args.cases,
        num_questions=args.num_questions,
        verbose=args.verbose,
    )
