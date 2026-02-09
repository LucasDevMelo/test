# Profectus AI Customer Support Agent

An intelligent customer support agent that answers questions using Profectus documentation and YouTube tutorials. The system uses retrieval-augmented generation (RAG) with Google ADK orchestration to provide accurate, cited responses.

## 🚀 Quick Start

**New to the project?** Start here:

1. **[Quick Start Guide](handover/07-QUICK-START.md)** - Get up and running in 15 minutes
2. **[Project Overview](handover/01-PROJECT-OVERVIEW.md)** - Understand the system architecture
3. **[Handover Documentation](handover/README.md)** - Complete documentation index

## 📋 What This Does

- **Evidence-based answers**: All responses cite sources from Help Center docs or YouTube videos
- **Smart escalation**: High-risk queries (billing, legal, account changes) are flagged for human review
- **Hybrid retrieval**: Combines keyword search (BM25) and semantic search (FAISS) for best results
- **Session management**: Maintains conversation context across multiple messages
- **Deep linking**: Provides direct links to relevant documentation and video timestamps

## 🏗️ Architecture

```
User Query
    │
    ▼
1. Classification (TRIVIAL / FACTUAL / HIGH_RISK)
    │
    ▼
2. Parallel Retrieval
    ├─ Keyword search (BM25)
    └─ Semantic search (FAISS)
    │
    ▼
3. Evidence Reader
    │
    ▼
4. Answer Generation (grounded in evidence)
    │
    ▼
5. Verification & Escalation
```

## 📂 Project Structure

```
Profectus-AI/
├── handover/              📚 Complete handover documentation (10 guides)
│   ├── README.md          Navigation and learning path
│   ├── 01-PROJECT-OVERVIEW.md
│   ├── 02-INGESTION-GUIDE.md
│   ├── 03-AGENT-ARCHITECTURE.md
│   ├── 04-API-INTEGRATION.md
│   ├── 05-MEMORY-SYSTEM.md
│   ├── 06-DEPLOYMENT-GUIDE.md
│   ├── 07-QUICK-START.md
│   ├── 08-INGESTION-TESTING.md       (11 verification tests)
│   ├── 09-VERIFICATION-RESULTS.md    (Feb 1, 2026 verification)
│   └── 10-FOLDER-STRUCTURE.md        (Complete directory guide)
│
├── backend/               🐍 Python backend
│   ├── profectus_ai/      Core RAG system
│   ├── Local_Final/       FAISS index + corpus (472 entries, verified ✅)
│   ├── requirements.txt   Dependencies (all tested ✅)
│   └── Dockerfile         Container build
│
├── frontend/              🌐 Web UI (static files)
│
├── scripts/               🔧 Deployment scripts
│   └── deploy_cloudrun.ps1
│
├── myenv/                 🐍 Virtual environment (Python 3.13.9, all deps installed)
│
└── archive/               📦 Development artifacts
    ├── openspec/          Development specs & tracking
    ├── dev-notes/         Code improvements, documentation updates
    ├── PROJECT_BREADCRUMBS.md
    └── OPERATIONS.md      Detailed ops commands
```

## 🎯 Common Tasks

### Run Locally
```powershell
# Activate environment
venv\Scripts\Activate.ps1

# Install dependencies
pip install -r backend\requirements.txt

# Set API key
$env:GOOGLE_API_KEY="your-key-here"

# Run server
cd backend
python -m uvicorn profectus_ai.api.app:app --port 8080 --reload
```

Visit: http://localhost:8080/

### Deploy to Cloud Run
```powershell
.\scripts\deploy_cloudrun.ps1
```

### Add New Content
```powershell
cd backend

# Fetch YouTube transcripts
python -m profectus_ai.get_transcript

# Process transcripts
python -m profectus_ai.process_youtube_timestamps

# Rebuild corpus
python -m profectus_ai.corpus_builder

# Rebuild index
python -m profectus_ai.build_faiss_index
```

See [02-INGESTION-GUIDE.md](handover/02-INGESTION-GUIDE.md) for details.

## 📊 Data Sources

| Source | Count | Location |
|--------|-------|----------|
| Help Center pages | 153 pages | `backend/profectus docs scraper/data/profectus_docs.sqlite` |
| Help Center entries | 330 chunks | After chunking (1200 chars/chunk) |
| YouTube videos | 26 videos | `backend/youtube/final_data_with_timestamps.json` |
| YouTube entries | 142 chunks | Full transcripts + time segments |
| **Total corpus** | **472 entries** | `backend/Local_Final/data.json` |

## 🔑 Environment Variables

### Required
```
GOOGLE_API_KEY=your-gemini-api-key
```

### Optional
```
PROFECTUS_SESSION_STORE=memory          # or firestore (production)
PROFECTUS_ADK_MODEL=gemini-2.5-flash    # LLM model
PROFECTUS_LOG_LEVEL=INFO                # Logging level
```

See [06-DEPLOYMENT-GUIDE.md](handover/06-DEPLOYMENT-GUIDE.md) for full configuration.

## 🛠️ Technology Stack

- **Agent Framework**: Google ADK (Agent Development Kit)
- **LLM**: Gemini 2.5 Flash
- **Vector Search**: FAISS
- **Embeddings**: all-MiniLM-L6-v2 (384-dim)
- **Keyword Search**: BM25 (rank-bm25)
- **API**: FastAPI + WebSocket
- **Deployment**: Google Cloud Run
- **Session Store**: Firestore (prod) or in-memory (dev)

## 📖 Documentation

**For new engineers**, follow this learning path:

1. **Day 1**: Setup and exploration
   - [Quick Start](handover/07-QUICK-START.md)
   - [Project Overview](handover/01-PROJECT-OVERVIEW.md)

2. **Day 2**: Architecture deep dive
   - [Agent Architecture](handover/03-AGENT-ARCHITECTURE.md)
   - [API Integration](handover/04-API-INTEGRATION.md)

3. **Day 3**: Operations
   - [Ingestion Guide](handover/02-INGESTION-GUIDE.md)
   - [Memory System](handover/05-MEMORY-SYSTEM.md)
   - [Deployment Guide](handover/06-DEPLOYMENT-GUIDE.md)

## 🔒 Safety & Escalation

The system automatically escalates queries involving:
- Billing and payments
- Account deletion or changes
- Legal matters
- Financial advice
- Explicit human agent requests

Escalations are logged to Firestore for human review.

## 🆘 Support

- **Setup issues**: See [Quick Start troubleshooting](handover/07-QUICK-START.md#troubleshooting)
- **Deployment issues**: See [Deployment Guide troubleshooting](handover/06-DEPLOYMENT-GUIDE.md#troubleshooting)
- **API issues**: See [API Integration](handover/04-API-INTEGRATION.md#error-handling)

## 🗂️ Archive

Development artifacts and historical docs are in [archive/](archive/):
- `archive/openspec/` - Development specs and change tracking
- `archive/dev-notes/` - Code improvement notes, documentation update logs
- `archive/OPERATIONS.md` - Detailed operational commands
- `archive/PROJECT_BREADCRUMBS.md` - Development history
- `backend/youtube/archive/` - Old YouTube processing versions

## 📝 Project Status

✅ **Production-ready and verified** (Feb 1, 2026)
- All dependencies installed and tested in `myenv`
- Complete ingestion pipeline verified end-to-end
- All 4 data stores/indexes verified:
  1. SQLite DB: 153 Help Center pages
  2. Unified Corpus: 472 entries with embeddings
  3. FAISS Index: 472 semantic vectors
  4. BM25 Index: 178 keyword entries
- Deployment script validated
- Cloud Run deployment active
- Comprehensive documentation (10 handover guides)

**See**: [09-VERIFICATION-RESULTS.md](handover/09-VERIFICATION-RESULTS.md) for full verification report

## 🚦 Getting Help

1. Check the [handover docs](handover/README.md)
2. Review the [troubleshooting sections](handover/07-QUICK-START.md#troubleshooting)
3. Check deployment logs: `gcloud logging read...`

---

**Ready to get started?** → [07-QUICK-START.md](handover/07-QUICK-START.md)
