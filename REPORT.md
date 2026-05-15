# Final Project Report

## 1. What & Why

StudyScribe is an AI-powered web application designed for students in lecture-heavy courses who need to transition from passive reading to active recall. While many students rely on re-reading highlighted notes, StudyScribe creates multiple-choice quizzes grounded strictly in the user's specific course materials. The app is built for students who need a personalized tutor capable of turning a messy lecture transcript or a set of dense PDF slides into an interactive flashcard experience. It is also helpful for students in lecture-heavy courses who need a way to turn passive notes into active recall practice.

The primary technical challenge in building StudyScribe was ensuring grounding. In a standard LLM interaction, the model often pulls from its vast base knowledge. But for a student, this is dangerous, if a professor defines a concept in a specific or non-standard way, the AI must use that specific definition, not the one it found in its training data. Getting the AI behavior right meant implementing a rigorous Retrieval-Augmented Generation (RAG) pipeline that forces the model to ignore its training data in favor of the provided context. This required deep iteration on system prompts and the implementation of OpenAI Function Calling to ensure the output remained structured enough for a web UI without sacrificing the academic accuracy of the content.

## 2. Iterations

### V1: Initial RAG Implementation

* **Change:** Established the baseline RAG pipeline. This included using `pypdf` for text extraction, `ChromaDB` for local vector storage, and a standard system prompt instructing the model to "be a helpful tutor" while generating a quiz.
* **Motivating Example:** During initial testing with a philosophy lecture, the AI correctly identified Thomas Hobbes but included facts about his other works (like *The Elements of Law*) that were never mentioned in the uploaded PDF.
* **Delta:** Question_Relevance_Score: 0.0 → 0.70.
* **Conclusion:** While the app successfully generated quizzes, the "helpfulness" of the AI was its weakness. It was supplementing the student's notes with outside information, which fails the requirement of being a localized study tool. 

### V2: Strict Terminology Enforcement

* **Change:** Updated the system prompt to include a "No Outside Knowledge" constraint and added Rule 5: "Use the EXACT terminology found in the context block." 
* **Motivating Example:** In a quiz about Hobbes, the AI referred to *The Leviathan* as a "book" or "work." While historically true, the source text referred to it as a "single absolute sovereign." V2 needed to reflect the text's specific phrasing to be considered accurate for the student's specific exam.
* **Delta:** Question_Relevance_Score: 0.70 → 0.85.
* **Conclusion:** Accuracy improved significantly. The model began using the specific phrasing found in the snippets. However, a new failure emerged: the model began over-generalizing logic, such as grouping different types of lecture "signals" into incorrect "All of the Above" answers.

### V3: Categorization Rigor & Dependency Resolution

* **Change:** Refined the prompt with Rule 6 (Categorization Rigor) to prevent logical hallucinations. 
* **Motivating Example:** The AI generated a question about "Lecture Signals" where it grouped "Now/Then" (time signals) and "For example" (example signals) into a single "All of the Above" answer. The source text explicitly categorized these differently.
* **Delta:** Question_Relevance_Score: 0.85 → 0.95.
* **Conclusion:** By adding strict categorization rules and fixing the underlying environment crashes, the AI finally achieved the "clinical" extraction needed for high-stakes study materials. The metric moved because the model stopped making logical leaps that were technically logical but contextually incorrect.

## 3. Code Walkthrough

The core of the application logic resides in `brain.py`, specifically the `generate_quiz` function (lines 160–220). When a user requests a quiz, the application first performs a semantic search against the local `ChromaDB` collection. On line 185, I use `collection.query(query_texts=[topic], n_results=6)` to retrieve the top six most relevant chunks of text. These chunks are then injected into a `system_prompt` as a context block.

A critical design decision was the use of **OpenAI Function Calling** (specifically the `tool_choice` parameter on line 206).

```python
tool_choice={"type": "function", "function": {"name": "generate_quiz_json"}}

```

By forcing the model to use this tool, I ensure that the LLM must return a valid JSON object matching my schema. An alternative I considered was simply asking for JSON in a standard text prompt. However, I rejected this because standard prompts often return conversational text (e.g., "Sure, here is your quiz:") or malformed JSON that would crash the frontend JavaScript's `JSON.parse()` method. Function calling guarantees that the UI can reliably iterate through `questions` and render them as interactive cards.

The frontend (`index.html`) handles the user action by sending the PDF to the `/upload` route in `app.py`. The backend stores the extracted text in a Flask `session` before passing it to `brain.py`. This separation ensures that the AI logic remains independent of the web framework, making the app easier to test via the `eval/run_evals.py` script.

## 4. AI Disclosure & Safety

I used an AI coding assistant (Kiro) to scaffold the initial Flask routes and the basic RAG structure. However, the assistant failed in three distinct moments that required manual recovery:

1. **Dependency Conflict:** The assistant generated a `requirements.txt` that did not pin the `numpy` version. This caused a crash on my macOS environment when NumPy 2.0 was released, as it was incompatible with `ChromaDB`. I recovered by manually identifying the error and pinning `numpy==1.26.4`.
2. **Logic Hallucination:** In the evaluation script, the assistant provided a fallback PDF generator that created malformed buffers. This led to a `negative seek value -1` error in `pypdf`. I recovered by identifying the need for the `reportlab` library and rewriting the `_fake_pdf` helper function.
3. **UI/UX Failure**: The initial website was a single-page upload tool with no user journey. To meet the project's UX requirements, added a sidebar for session history and a "Concepts to Review" dashboard, transforming it into a proper study application.
4. **No session progress**: The AI assistant initially treated every quiz as a single event. It would generate questions but had no memory of what the student had already answered, meaning it couldn't fulfill the 'session history' requirement promised in my proposal. I modified `app.py` to utilize the Flask session to store a running dictionary of student performance. 
5. **No review**: The original code provided no feedback loop. It could generate a quiz but could not identify which concepts a student was failing. I engineered the weak_concepts algorithm in brain.py to keep a tally of incorrect answers and surface specific topics

**Safety Risk & Mitigation:** A significant safety risk for StudyScribe is PII (Personally Identifiable Information) Exposure. If a student uploads a graded paper containing their name, ID number, or professor’s contact info, those could potentially be stored in the local vector database. To mitigate this, the app uses a local client for ChromaDB. No data is ever sent to a third-party vector cloud (like Pinecone), the only external transit is to OpenAI's API for processing, which is governed by their enterprise privacy policy for API data. 