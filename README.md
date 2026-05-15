# cpsc254-final-project-jadethong

# StudyScribe

StudyScribe is an AI-powered study assistant that transforms lecture notes and PDFs into interactive multiple-choice quizzes. Utilizing Retrieval-Augmented Generation (RAG), it ensures that every question is strictly grounded in the uploaded course materials, preventing the "hallucinations" often found in generic AI tools.

### Key Features

* **Context-Aware Quiz Generation:** Uses a local vector database (ChromaDB) to retrieve specific lecture segments before generating questions.
* **Session Progress Tracking:** Monitors student performance across different concepts throughout a study session.
* **"Topics to Review" Dashboard:** Automatically identifies and surfaces weak areas where accuracy is below 60%.
* **Strict RAG Guardrails:** Prevents the AI from using outside training data, ensuring the quiz reflects the professor's specific definitions (e.g., distinguishing between Hobbes' and Rousseau's views on the Social Contract).

---

### Prerequisites

* **Python:** 3.10 or higher
* **OpenAI API Key:** Required for quiz generation and embeddings.

---

### Setup Instructions

1. **Clone the Repository**
```bash
git clone https://github.com/jadethong/cpsc254-final-project-jadethong.git
cd cpsc254-final-project-jadethong

```


2. **Create and Activate a Virtual Environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```


3. **Install Dependencies**
*Note: This includes a specific version of NumPy to ensure compatibility with the local vector store.*
```bash
pip install -r requirements.txt

```


4. **Configure Environment Variables**
Create a `.env` file in the root directory and add your OpenAI API key:
```bash
echo "OPENAI_API_KEY=your_actual_key_here" > .env

```



---

### How to Run

#### 1. Start the Web Application

Launch the Flask server:

```bash
python app.py

```

* Open your browser to `http://127.0.0.1:5000`.
* Upload a PDF (e.g., `lecture_philosophy.pdf`).
* Select the number of questions and click **Generate Quiz**.

#### 2. Run the Evaluation Suite

To verify the accuracy and grounding of the RAG pipeline using the automated test harness:

```bash
python eval/run_evals.py --verbose

```

* **Case 1:** Tests in-scope retrieval (TCP vs UDP).
* **Case 2:** Tests out-of-scope refusal (Attempting to ask about the Roman Empire using RSA encryption notes).

---

### Project Structure

* `app.py`: Flask backend and session management.
* `brain.py`: Core AI logic, RAG pipeline, and vector database management.
* `eval/`: Contains `test_cases.json` and the evaluation script.
* `static/` & `templates/`: Frontend UI assets and interactive flashcard logic.
* `requirements.txt`: Pinned dependencies including `reportlab` for evaluation PDF generation and `numpy==1.26.4`.
