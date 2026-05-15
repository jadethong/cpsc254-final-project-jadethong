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

---

### Project Structure

* **`app.py`**: The Flask backend that manages routing, PDF uploads, and the persistent user session.
* **`brain.py`**: The core AI logic. This contains the RAG pipeline, ChromaDB vectorization, and the OpenAI Function Calling implementation for quiz generation.
* **`eval/`**:
* **`test_cases.json`**: A comprehensive suite of 10 labeled test cases covering in-scope and out-of-scope scenarios.
* **`run_evals.py`**: The evaluation harness used to calculate the `Question_Relevance_Score`.
* **`corpus/`**: Contains sample lecture PDFs (such as `lecture_philosophy.pdf`) used for RAG testing and demonstration.
* **`index.html`**: The interactive frontend UI, featuring the upload zone and the flashcard study interface.
* **`chroma_db/`**: A local directory where the vector embeddings and document chunks are stored persistently using ChromaDB.
* **`requirements.txt`**: The complete list of pinned dependencies for database compatibility.
* **`REPORT.md`**: The technical project report detailing iterations, AI behavior depth, and disclosure of AI assistant usage.
