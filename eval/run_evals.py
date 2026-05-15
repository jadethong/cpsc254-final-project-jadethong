"""
eval/run_evals.py — StudyScribe evaluation harness.

Metric
------
Question_Relevance_Score = (number of generated questions where the correct
    answer exists within the retrieved context) / (total number of generated
    quiz questions)

For out-of-scope cases the formula becomes:
    Score = 1.0  if model correctly sets out_of_scope=True with questions=[]
                 and a non-empty refusal_message
    Score = 0.0  if model generates any questions for an off-topic prompt

10 test cases are run:
    IDs 1-8   expected_type = "in_scope"   — structured JSON + answer-in-context
    IDs 9-10  expected_type = "out_of_scope" — refusal required, no questions

Usage
-----
    cd <project-root>
    python eval/run_evals.py [--cases eval/test_cases.json] [--verbose]

Passing threshold: every case must score > 0.0
Overall score reported as mean Question_Relevance_Score across all 10 cases.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from brain import index_documents, generate_quiz


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

REQUIRED_KEYS = {"question", "choices", "answer", "explanation", "concept"}
PASS_THRESHOLD = 0.0   # every case must score strictly above this


# ─────────────────────────────────────────────────────────────────────────────
# Minimal in-memory PDF builder
# (avoids needing real PDF files on disk for evals)
# ─────────────────────────────────────────────────────────────────────────────

def _fake_pdf(text: str) -> bytes:
    """
    Build a minimal valid PDF in memory that contains `text`.
    Tries reportlab first (richer encoding); falls back to a raw PDF skeleton.
    """
    import io
    try:
        from reportlab.pdfgen import canvas as rl_canvas
        buf = io.BytesIO()
        c = rl_canvas.Canvas(buf)
        y = 750
        for line in text.splitlines():
            for chunk in [line[i:i + 90] for i in range(0, max(len(line), 1), 90)]:
                c.drawString(40, y, chunk)
                y -= 14
                if y < 50:
                    c.showPage()
                    y = 750
        c.save()
        return buf.getvalue()
    except ImportError:
        # Raw minimal PDF — ChromaDB's pypdf can still extract the text
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        lines: list[str] = []
        for raw in safe.splitlines():
            while len(raw) > 80:
                lines.append(raw[:80])
                raw = raw[80:]
            lines.append(raw)
        stream_lines = "\n".join(
            f"BT /F1 10 Tf 40 {750 - i * 14} Td ({ln}) Tj ET"
            for i, ln in enumerate(lines[:50])
        )
        stream = stream_lines.encode()
        header = (
            "%PDF-1.4\n"
            "1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n"
            "2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj\n"
            "3 0 obj<</Type/Page/MediaBox[0 0 612 792]/Parent 2 0 R"
            "/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>endobj\n"
            f"4 0 obj<</Length {len(stream)}>>\nstream\n"
        ).encode()
        footer = (
            b"\nendstream\nendobj\n"
            b"5 0 obj<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>endobj\n"
            b"xref\n0 6\n"
            b"0000000000 65535 f \n"
            b"trailer<</Size 6/Root 1 0 R>>\n"
            b"startxref\n0\n%%EOF\n"
        )
        return header + stream + footer


# ─────────────────────────────────────────────────────────────────────────────
# Scoring helpers
# ─────────────────────────────────────────────────────────────────────────────

def _strip_label(answer: str) -> str:
    """Remove leading choice label ('A. ', 'B. ', etc.) from an answer string."""
    if len(answer) > 2 and answer[1] == "." and answer[0].upper() in "ABCD":
        return answer[3:].strip()
    return answer.strip()


def answer_in_context(answer: str, context: str) -> bool:
    """
    Return True when the correct answer can be traced back to the context.
    Strategy: at least one 'meaningful' word (≥4 chars) from the stripped
    answer must appear in the context (case-insensitive).
    """
    ctx = context.lower()
    stripped = _strip_label(answer)
    words = [w.strip(".,;:!?\"'()") for w in stripped.split()]
    meaningful = [w for w in words if len(w) >= 4]
    return any(w.lower() in ctx for w in meaningful)


def validate_question(q: dict, context: str) -> tuple[bool, list[str]]:
    """
    Check a single question dict against four criteria.
    Returns (passed: bool, failures: list[str]).
    """
    failures: list[str] = []

    # 1. All required keys present
    missing = REQUIRED_KEYS - set(q.keys())
    if missing:
        failures.append(f"missing keys: {sorted(missing)}")

    # 2. Exactly 4 choices
    choices = q.get("choices", [])
    if len(choices) != 4:
        failures.append(f"expected 4 choices, got {len(choices)}")

    # 3. Answer matches a choice exactly
    answer = q.get("answer", "")
    if answer not in choices:
        failures.append(f"answer '{answer}' not found in choices")

    # 4. Answer is grounded in the retrieved context
    if not answer_in_context(answer, context):
        failures.append(f"answer not traceable to context: '{answer}'")

    return (len(failures) == 0, failures)


def bar(score: float, width: int = 24) -> str:
    filled = int(round(score * width))
    return "[" + "█" * filled + "░" * (width - filled) + f"] {score:.0%}"


# ─────────────────────────────────────────────────────────────────────────────
# Generic case runner  (handles BOTH in_scope and out_of_scope)
# ─────────────────────────────────────────────────────────────────────────────

def run_case(case: dict, verbose: bool) -> tuple[float, list[str]]:
    """
    Run a single eval case and return (score, notes).

    in_scope cases
    --------------
    Score = grounded_questions / total_questions   (0.0 – 1.0)
    A question is 'grounded' when validate_question() returns True.

    out_of_scope cases
    ------------------
    Score = 1.0  iff  out_of_scope=True  AND  questions=[]  AND refusal_message != ""
    Score = 0.0  otherwise
    """
    cid      = case["id"]
    context  = case["context"]
    topic    = case["topic"]
    n        = case["num_questions"]
    etype    = case["expected_type"]
    notes: list[str] = []

    # ── Index context into ChromaDB ──
    fname = f"eval_case_{cid}.pdf"
    index_documents([(fname, _fake_pdf(context))])

    # ── Call the AI ──
    result = generate_quiz(topic=topic, num_questions=n)

    if verbose:
        print("\n  Raw model output:")
        print(json.dumps(result, indent=4))

    out_of_scope  = result.get("out_of_scope", False)
    questions     = result.get("questions", [])
    refusal_msg   = (result.get("refusal_message") or "").strip()

    # ── Score: out-of-scope case ──
    if etype == "out_of_scope":
        if out_of_scope and len(questions) == 0 and refusal_msg:
            notes.append("  ✅ PASS — model correctly refused out-of-scope topic")
            notes.append(f'  Refusal: "{refusal_msg}"')
            return 1.0, notes

        if not out_of_scope:
            notes.append("  ❌ out_of_scope flag was False — model did not refuse")
        if questions:
            notes.append(f"  ❌ model generated {len(questions)} question(s) for an off-topic prompt")
        if not refusal_msg:
            notes.append("  ❌ refusal_message is empty")
        return 0.0, notes

    # ── Score: in-scope case ──
    if out_of_scope:
        notes.append("  ❌ FAIL — model marked in-scope content as out_of_scope")
        return 0.0, notes

    if not questions:
        notes.append("  ❌ FAIL — no questions returned")
        return 0.0, notes

    passed = 0
    for i, q in enumerate(questions, 1):
        ok, failures = validate_question(q, context)
        if ok:
            passed += 1
            notes.append(f"  Q{i} ✅  concept: '{q.get('concept', '')}'")
        else:
            notes.append(f"  Q{i} ❌  {'; '.join(failures)}")
        if verbose:
            notes.append(f"       Q: {q.get('question', '')}")
            notes.append(f"       A: {q.get('answer', '')}")

    score = passed / len(questions)
    return score, notes


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_evals(cases_path: str, verbose: bool) -> None:
    with open(cases_path, encoding="utf-8") as f:
        test_cases: list[dict] = json.load(f)

    print("=" * 62)
    print("  StudyScribe — Evaluation Report")
    print(f"  Metric : Question_Relevance_Score")
    print(f"  Cases  : {len(test_cases)}  "
          f"({sum(1 for c in test_cases if c['expected_type']=='in_scope')} in-scope, "
          f"{sum(1 for c in test_cases if c['expected_type']=='out_of_scope')} out-of-scope)")
    print("=" * 62)

    results: list[dict] = []
    total_start = time.time()

    for case in test_cases:
        cid  = case["id"]
        name = case["name"]
        print(f"\n{'─' * 62}")
        print(f"  Case {cid:02d}: {name}")
        print(f"  Type   : {case['expected_type']}")
        print(f"  Topic  : {case['topic']}")

        start = time.time()
        try:
            score, notes = run_case(case, verbose)
        except Exception as exc:
            score = 0.0
            notes = [f"  ❌ EXCEPTION: {exc}"]
        elapsed = time.time() - start

        for note in notes:
            print(note)

        if case["expected_type"] == "out_of_scope":
            status = "✅ PASS" if score == 1.0 else "❌ FAIL"
        else:
            status = "✅ PASS" if score == 1.0 else ("⚠️  PARTIAL" if score > 0 else "❌ FAIL")

        print(f"\n  Question_Relevance_Score : {bar(score)}  {status}  ({elapsed:.1f}s)")
        results.append({"id": cid, "name": name, "type": case["expected_type"], "score": score})

    # ── Summary table ─────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    overall = sum(r["score"] for r in results) / len(results) if results else 0.0

    in_scope_scores = [r["score"] for r in results if r["type"] == "in_scope"]
    oos_scores      = [r["score"] for r in results if r["type"] == "out_of_scope"]

    print(f"\n{'=' * 62}")
    print("  SUMMARY")
    print(f"{'=' * 62}")
    print(f"  Cases run          : {len(results)}")
    print(f"  Total time         : {total_elapsed:.1f}s")
    print()
    print(f"  Overall  Question_Relevance_Score : {bar(overall)}")
    if in_scope_scores:
        avg_in = sum(in_scope_scores) / len(in_scope_scores)
        print(f"  In-scope avg score                : {bar(avg_in)}")
    if oos_scores:
        avg_oos = sum(oos_scores) / len(oos_scores)
        print(f"  Out-of-scope refusal rate         : {bar(avg_oos)}")
    print()

    # Per-case rows
    col_w = 42
    print(f"  {'ID':<4} {'Type':<13} {'Score':>6}  {'Pass?':<6}  Name")
    print(f"  {'─'*4} {'─'*13} {'─'*6}  {'─'*6}  {'─'*col_w}")
    for r in results:
        if r["type"] == "out_of_scope":
            flag = "✅" if r["score"] == 1.0 else "❌"
        else:
            flag = "✅" if r["score"] == 1.0 else ("⚠️ " if r["score"] > 0 else "❌")
        print(f"  {r['id']:<4} {r['type']:<13} {r['score']:>5.0%}   {flag}     {r['name'][:col_w]}")

    print("=" * 62)

    # ── Exit code ─────────────────────────────────────────────────────────
    failures = [r for r in results if r["score"] <= PASS_THRESHOLD]
    if failures:
        print(f"\n⚠️  {len(failures)} case(s) scored 0. See details above.")
        sys.exit(1)
    else:
        print(f"\n✅ All {len(results)} cases passed  (overall {overall:.0%}).")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run StudyScribe evals.")
    parser.add_argument(
        "--cases",
        default=os.path.join(os.path.dirname(__file__), "test_cases.json"),
        help="Path to test_cases.json (default: eval/test_cases.json)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print the raw model output for every case",
    )
    args = parser.parse_args()
    run_evals(cases_path=args.cases, verbose=args.verbose)
