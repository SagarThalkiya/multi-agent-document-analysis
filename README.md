## Multi-Agent Document Analysis System

A FastAPI-based multi-agent document analysis system that processes PDF and TXT files using three specialized AI agents working in **true parallel execution**. The system provides structured insights including summaries, entity extraction, and sentiment analysis.

### Demo
- [Project walkthrough (Loom)](https://www.loom.com/share/6366581ea59a45b483a224ddad50e464)

### Features
- **True Parallel Agent Execution** - All three agents execute simultaneously using `asyncio.gather()`, ensuring total time ≈ slowest agent (not sum of all agents)
- **LLM-Powered Analysis** - Integrated with Groq API (free tier) using LangChain for intelligent document processing
- **Heuristic Fallback** - Graceful degradation to rule-based processing when LLM is unavailable
- **Async Background Processing** - FastAPI `BackgroundTasks` for non-blocking API responses
- **Robust Job Tracking** - In-memory state management with per-agent timing, completion counts, and error handling
- **PDF/TXT Support** - Document parsing with `pdfplumber` and size/type validation (max 10MB)
- **Comprehensive Testing** - 28 unit and integration tests covering all agents and parallel execution verification
- **HTML/CSS/JavaScript Frontend** - Clean, functional UI for document upload, analysis triggering, and results display

### Prerequisites
- Python 3.10+
- Groq API key (free tier available at https://console.groq.com/) - Optional but recommended
- Node is *not* required (frontend is static HTML/JS)

### Backend Setup

1. **Create and activate virtual environment:**
```bash
cd backend
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # macOS/Linux
```

2. **Install dependencies:**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3. **Configure Groq API key (optional but recommended):**
```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Groq API key:
# GROQ_API_KEY=your_groq_api_key_here
```

**Note:** Without a Groq API key, the system will use heuristic-based processing (still functional but less intelligent).

4. **Run the API server:**
```bash
# From project root
backend\.venv\Scripts\uvicorn app.main:app --reload --app-dir backend
```

The API will be available at `http://127.0.0.1:8000`

### Frontend Setup

Serve the static files (in a separate terminal):
```bash
# From project root
python -m http.server 5500 --directory frontend
```

Open your browser and visit: `http://localhost:5500`

**Alternative:** You can also use VS Code "Live Server" extension or any static file server.

### API Endpoints

1. **POST /upload** - Upload a document (PDF or TXT, max 10MB)
   - Returns: `job_id`, `filename`, `status`, `message`
   
2. **POST /analyze** - Trigger multi-agent analysis
   - Body: `{"job_id": "abc123"}`
   - Returns: `job_id`, `status: "processing"`, `message`
   - Starts all three agents in parallel immediately

3. **GET /results/{job_id}** - Retrieve analysis results
   - Returns: Complete or partial results with per-agent timing
   - Status: `completed`, `partial`, or `processing`

### Example API Usage

```bash
# 1. Upload a document
curl -X POST http://localhost:8000/upload \
  -F "file=@sample_documents/sample.txt"

# Response: {"job_id": "abc123", "filename": "sample.txt", "status": "uploaded", ...}

# 2. Start analysis
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"job_id": "abc123"}'

# Response: {"job_id": "abc123", "status": "processing", "message": "Analysis started..."}

# 3. Get results (poll until status is "completed" or "partial")
curl http://localhost:8000/results/abc123

# Response: Complete results with summary, entities, sentiment, and timing data
```

### Testing

The project includes comprehensive unit tests for all agents and parallel execution verification:

```bash
# Run all tests (28 tests total)
backend\.venv\Scripts\python -m pytest backend\test_agents.py backend\test_api.py backend\test_parallel_execution.py -v

# Run only agent unit tests
backend\.venv\Scripts\python -m pytest backend\test_agents.py -v

# Run parallel execution verification tests
backend\.venv\Scripts\python -m pytest backend\test_parallel_execution.py -v

# Run integration tests
backend\.venv\Scripts\python -m pytest backend\test_api.py -v

**Total: 10 tests, all passing**

### Design Decisions

- **Agent framework (LangChain):** I chose LangChain because it provides structured prompt templates (`ChatPromptTemplate`) and a simple composition pattern (`prompt | ChatGroq`) that keeps each agent’s logic declarative. It also let me swap between Groq-backed LLMs and heuristic fallbacks without rewriting business logic.
- **Frontend framework (vanilla HTML/JS):** The assessment only needs a lightweight proof-of-concept UI, so plain HTML/CSS/JS keeps the footprint tiny, avoids bundler setup, and makes it easy for reviewers to inspect the fetch calls hitting the FastAPI endpoints.
- **Parallel execution:** The orchestrator wraps each agent call in `_timed()` and launches them with `asyncio.gather()` (see `services/orchestrator.py`). This ensures the total wall-clock time is bounded by the slowest agent rather than the sum; parallelism is verified in `test_parallel_execution.py`.
- **Agent failure handling:** Any exception inside an agent is caught, recorded in an `AgentOutcome`, and surfaced as a partial response (status `partial`) while successful agents still return data. This prevents a single LLM/network failure from sinking the entire job.
- **LLM provider (Groq):** Groq’s free tier is fast and generous, and the `langchain-groq` package made integration trivial. If `GROQ_API_KEY` is absent, the system automatically falls back to deterministic regex-based heuristics.
- **Challenges & fixes:** 
  1. **Groq model deprecation** (llama-3.1-70b retired) caused 400 errors—resolved by upgrading to `llama-3.3-70b` and adding diagnostic logging.
  2. **LLM JSON parsing**—Groq sometimes wraps JSON in ``` fences; I added code to strip code blocks and balance braces before `json.loads`.
  3. **Async testing**—pytest needs `pytest-asyncio`; adding it to `requirements.txt` fixed the “async def not supported” errors.

### Architecture & Implementation

#### Multi-Agent System
- **SummarizerAgent** (`backend/app/agents/summarizer.py`)
  - Generates concise summaries (max 150 words)
  - Extracts 3-5 key points
  - Returns confidence scores
  - Supports LLM (Groq) and heuristic fallback modes

- **EntityExtractorAgent** (`backend/app/agents/entity_extractor.py`)
  - Extracts people, organizations, dates, and locations
  - Counts entity mentions
  - Provides context for each entity
  - Supports LLM (Groq) and heuristic fallback modes

- **SentimentAgent** (`backend/app/agents/sentiment.py`)
  - Determines tone (positive/negative/neutral)
  - Calculates confidence scores (0-1)
  - Identifies formality level
  - Extracts supporting key phrases
  - Supports LLM (Groq) and heuristic fallback modes

#### Parallel Execution
The orchestrator (`backend/app/services/orchestrator.py`) uses `asyncio.gather()` to execute all three agents **truly in parallel**:

```python
# All agents start simultaneously
raw_results_dict = await asyncio.gather(
    self._run_summary(document_text),
    self._run_entities(document_text),
    self._run_sentiment(document_text),
    return_exceptions=True
)
```

**Performance:** Total time ≈ max(agent_times), not sum(agent_times)
- Example: If agents take 4s, 5s, 3s → Total time ≈ 5s (not 12s)

#### Key Components
- **Orchestrator** - Coordinates parallel execution, measures timing, handles errors gracefully
- **Document Parser** - Centralized TXT/PDF extraction with error handling
- **Job Management** - In-memory tracking with status, timing, and partial result support
- **Frontend** - Vanilla JS with real-time status updates and result display

### Project Structure

```
multi-agent-doc/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application & endpoints
│   │   ├── models.py            # Pydantic request/response models
│   │   ├── agents/              # Three specialized agents
│   │   │   ├── summarizer.py
│   │   │   ├── entity_extractor.py
│   │   │   └── sentiment.py
│   │   ├── services/
│   │   │   └── orchestrator.py  # Parallel execution coordinator
│   │   └── utils/
│   │       └── document_parser.py
│   ├── test_agents.py           # Unit tests for all agents
│   ├── test_api.py              # Integration tests
│   ├── test_parallel_execution.py  # Parallel execution verification
│   ├── requirements.txt
│   ├── .env.example
│   └── uploads/                 # Stored documents
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── styles.css
├── sample_documents/
│   ├── sample.txt
│   └── sample.pdf
└── README.md
```

### Performance Characteristics

- **Small documents (<100KB):** ~0.5-2 seconds
- **Medium documents (100KB-1MB):** ~2-5 seconds
- **Large documents (1MB-10MB):** ~5-15 seconds (depends on content complexity)

**Note:** With Groq LLM enabled, processing is faster and more accurate. Heuristic mode is slower for large documents but still functional.

### Troubleshooting

**Issue: "GROQ_API_KEY not set"**
- Solution: Create `backend/.env` with `GROQ_API_KEY=your_key` (optional - system works without it)

**Issue: 400 Bad Request from Groq**
- Solution: Check that your API key is valid and the model name is correct (uses `llama-3.3-70b-versatile`)

**Issue: Slow processing for large files**
- Solution: Ensure agents are running in parallel (check logs show simultaneous execution). The system is optimized for parallel execution.

**Issue: Tests failing**
- Solution: Ensure `pytest` and `pytest-asyncio` are installed: `pip install pytest pytest-asyncio`

### Future Enhancements
- [ ] Support for additional LLM providers (OpenAI, Gemini, Anthropic)
- [ ] Persistent job storage (Redis/PostgreSQL) for multi-instance deployments
- [ ] Real-time progress streaming via WebSockets
- [ ] Enhanced NLP with spaCy/transformers for better entity extraction
- [ ] Batch document processing
- [ ] Export results to various formats (JSON, CSV, PDF)
