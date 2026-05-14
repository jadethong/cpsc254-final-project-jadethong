"""
eval/run_evals.py — StudyScribe evaluation harness.

Metric
------
Question_Relevance_Score = (number of generated questions where the correct
    answer exists within the retrieved context) / (total number of generated
    quiz questions)

For the out-of-scope test case the formula becomes:
    Score = 1.0  if the model correctly refuses (out_of_scope=True, questions=[])
    Score = 0.0  if the model generates any questions for an off-topic prompt

Two test cases are evaluated:
  Case 1 — TCP vs UDP (in-scope): index the TCP/UDP context, generate 2 questions,
            verify JSON structure + answer-in-context for each question.
  Case 2 — Roman Empire (out-of-scope): index RSA context, request a quiz about
            the Roman Empire, verify the model returns out_of_scope=True with no
            questions.

Usage
-----
    cd <project-root>
    python eval/run_evals.py [--verbose]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from brain import index_documents, generate_quiz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = {"question", "choices", "answer", "explanation", "concept"}


def _fake_pdf(text: str) -> bytes:
    """
    Create a minimal valid PDF in memory containing `text`.
    This lets us reuse brain.index_documents() without needing real PDF files.
    """
    import io
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        # wrap long text across lines
        y = 750
        for line in text.splitlines():
            for chunk in [line[i:i+90] for i in range(0, max(len(line),1), 90)]:
                c.drawString(40, y, chunk)
                y -= 14
                if y < 50:
                    c.showPage()
                    y = 750
        c.save()
        return buf.getvalue()
    except ImportError:
        # reportlab not installed — build a raw minimal PDF with the text embedded
        content = (
            "%PDF-1.4\n"
            "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            "3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
            "/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
        )
        # Encode text as PDF stream
        safe_text = text.replace("\\","\\\\").replace("(","\\(").replace(")","\\)")
        # Split into 80-char display lines
        lines = []
        for raw in safe_text.splitlines():
            while len(raw) > 80:
                lines.append(raw[:80])
                raw = raw[80:]
            lines.append(raw)
        stream_lines = "\n".join(f"BT /F1 10 Tf 40 {750 - i*14} Td ({ln}) Tj ET"
                                  for i, ln in enumerate(lines[:50]))
        stream = stream_lines.encode()
        content += (
            f"4 0 obj<</Length {len(stream)}>>\nstream\n"
        )
        content = content.encode() + stream + b"\nendstream\nendobj\n"
        content += (
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"trailer<</Size 6/Root 1 0 R>>\n"
            b"startxref\n0\n%%EOF\n"
        )
        return content


def answer_in_context(answer: str, context: str) -> bool:
    """Return True if any meaningful word from `answer` appears in `context`."""
    ctx_lower = context.lower()
    # Strip choice label (e.g. "A. ") before checking
    ans = answer
    if len(ans) > 2 and ans[1] == "." and ans[0].upper() in "ABCD":
        ans = ans[3:].strip()
    # Check for any 4+ character word overlap
    words = [w.strip(".,;:!?\"'") for w in ans.split() if len(w.strip(".,;:!?\"'")) >= 4]
    return any(w.lower() in ctx_lower for w in words)


def validate_question(q: dict, context: str) -> tuple[bool, list[str]]:
    """
    Returns (passed, list_of_failures).
    A question passes when:
      1. All required keys are present.
      2. choices has exactly 4 entries.
      3. answer matches one of the choices exactly.
      4. answer (stripped of label) has at least one word in the context.
    """
    failures = []

    # 1. Required keys
    missing = REQUIRED_KEYS - set(q.keys())
    if missing:
        failures.append(f"Missing keys: {missing}")

    # 2. Four choices
    choices = q.get("choices", [])
    if len(choices) != 4:
        failures.append(f"Expected 4 choices, got {len(choices)}")

    # 3. Answer in choices
    answer = q.get("answer", "")
    if answer not in choices:
        failures.append(f"answer '{answer}' not found in choices")

    # 4. Answer grounded in context
    if not answer_in_context(answer, context):
        failures.append(f"answer '{answer}' cannot be traced to the provided context")

    return (len(failures) == 0, failures)


def bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


# ---------------------------------------------------------------------------
# Individual case runners
# ---------------------------------------------------------------------------

def run_case_1(case: dict, verbose: bool) -> tuple[float, list[str]]:
    """
    Case 1: in-scope TCP vs UDP.
    Question_Relevance_Score = questions_with_answer_in_context / total_questions
    """
    context = case["context"]
    topic   = case["topic"]
    n       = case["num_questions"]
    crit    = case["passing_criteria"]
    notes   = []

    print(f"\n  Indexing context for Case {case['id']}…")
    index_documents([("tcp_udp_lecture.pdf", _fake_pdf(context))])

    print(f"  Calling generate_quiz(topic='{topic}', num_questions={n})…")
    result = generate_quiz(topic=topic, num_questions=n)

    if verbose:
        print("\n  Raw result:")
        print(json.dumps(result, indent=4))

    # Must NOT be out of scope
    if result.get("out_of_scope"):
        notes.append("FAIL: model marked in-scope content as out_of_scope")
        return 0.0, notes

    questions = result.get("questions", [])
    if not questions:
        notes.append("FAIL: no questions returned")
        return 0.0, notes

    # Evaluate each question
    passed_count = 0
    for i, q in enumerate(questions, 1):
        ok, failures = validate_question(q, context)
        if ok:
            passed_count += 1
            notes.append(f"  Q{i} ✅ PASS — concept: '{q.get('concept','')}'")
        else:
            notes.append(f"  Q{i} ❌ FAIL — {'; '.join(failures)}")
        if verbose:
            notes.append(f"       question : {q.get('question','')}")
            notes.append(f"       answer   : {q.get('answer','')}")

    score = passed_count / len(questions)
    return score, notes


def run_case_2(case: dict, verbose: bool) -> tuple[float, list[str]]:
    """
    Case 2: out-of-scope refusal.
    Score = 1.0 if out_of_scope=True AND questions=[].
    Score = 0.0 otherwise.
    """
    context = case["context"]
    topic   = case["topic"]
    n       = case["num_questions"]
    notes   = []

    print(f"\n  Indexing context for Case {case['id']} (RSA content)…")
    index_documents([("rsa_lecture.pdf", _fake_pdf(context))])

    print(f"  Calling generate_quiz(topic='{topic}', num_questions={n})…")
    result = generate_quiz(topic=topic, num_questions=n)

    if verbose:
        print("\n  Raw result:")
        print(json.dumps(result, indent=4))

    out_of_scope = result.get("out_of_scope", False)
    questions    = result.get("questions", [])
    refusal_msg  = result.get("refusal_message", "").strip()

    if out_of_scope and len(questions) == 0 and refusal_msg:
        notes.append("  ✅ PASS — model correctly refused out-of-scope topic")
        notes.append(f"  Refusal message: \"{refusal_msg}\"")
        return 1.0, notes

    if not out_of_scope:
        notes.append("  ❌ FAIL — model did NOT set out_of_scope=True")
    if questions:
        notes.append(f"  ❌ FAIL — model generated {len(questions)} question(s) for an off-topic prompt")
    if not refusal_msg:
        notes.append("  ❌ FAIL — refusal_message is empty")

    return 0.0, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_evals(cases_path: str, verbose: bool) -> None:
    with open(cases_path, encoding="utf-8") as f:
        test_cases = json.load(f)

    print("=" * 60)
    print("  StudyScribe — Evaluation Report")
    print(f"  Metric: Question_Relevance_Score")
    print("=" * 60)

    runners = {1: run_case_1, 2: run_case_2}
    results = []
    total_start = time.time()

    for case in test_cases:
        cid  = case["id"]
        name = case["name"]
        print(f"\n{'─'*60}")
        print(f"  Case {cid}: {name}")
        print(f"  Expected type: {case['expected_type']}")

        start = time.time()
        runner = runners.get(cid)
        if runner is None:
            print(f"  ⚠️  No runner for case {cid}, skipping.")
            continue

        try:
            score, notes = runner(case, verbose)
            elapsed = time.time() - start
        except Exception as e:
            elapsed = time.time() - start
            score, notes = 0.0, [f"  ❌ EXCEPTION: {e}"]

        for note in notes:
            print(note)

        status = "✅ PASS" if score >= 1.0 else ("⚠️  PARTIAL" if score > 0 else "❌ FAIL")
        print(f"\n  Question_Relevance_Score: {bar(score)}  {status}  ({elapsed:.1f}s)")
        results.append({"id": cid, "name": name, "score": score})

    # ── Summary ──────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  Cases run  : {len(results)}")
    print(f"  Total time : {total_elapsed:.1f}s")
    print()
    print(f"  Overall Question_Relevance_Score: {bar(overall)}")
    print()
    print(f"  {'ID':<5} {'Score':>6}  Name")
    print(f"  {'─'*5} {'─'*6}  {'─'*44}")
    for r in results:
        flag = "✅" if r["score"] >= 1.0 else ("⚠️ " if r["score"] > 0 else "❌")
        print(f"  {r['id']:<5} {r['score']:>5.0%}   {flag} {r['name']}")
    print("=" * 60)

    # Non-zero exit if any case fully fails
    if any(r["score"] == 0.0 for r in results):
        print("\n⚠️  One or more cases scored 0. See details above.")
        sys.exit(1)
    else:
        print(f"\n✅ All cases passed (overall {overall:.0%}).")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StudyScribe evals.")
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases.json"),
        help="Path to test_cases.json",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print raw model output for each case",
    )
    args = parser.parse_args()
    run_evals(cases_path=args.cases, verbose=args.verbose)
