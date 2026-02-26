# 🎓 KARE FAQ Chatbot

An AI-powered FAQ chatbot for **Kalasalingam Academy of Research and Education (KARE)** built with semantic search and Retrieval-Augmented Generation (RAG).

## ✨ Features

- **Semantic Search** — Uses `all-MiniLM-L6-v2` sentence-transformer embeddings for meaning-based retrieval (not keyword matching)
- **RAG Pipeline** — Retrieved FAQ entries are passed to Google Gemini (`gemini-2.0-flash`) for natural, comprehensive answers
- **Dynamic Retrieval** — Overview queries automatically retrieve more entries (8) for broader answers; specific queries retrieve 5
- **Graceful Fallback** — If Gemini is unavailable, the best FAQ match is returned directly
- **300+ FAQ Entries** across 11 categories (Admissions, Academics, Placements, Campus Life, etc.)

## 🏗️ Architecture

```
college-faq-chatbot/
├── backend/                  # Flask API server
│   ├── app.py                # Entry point & app factory
│   ├── config.py             # Centralised configuration
│   ├── kare_faq.json         # FAQ dataset (300 entries)
│   ├── requirements.txt
│   ├── routes/
│   │   └── chat.py           # POST /api/chat endpoint
│   ├── services/
│   │   ├── matching_engine.py  # Semantic search engine
│   │   └── llm_generator.py    # Gemini RAG wrapper
│   └── utils/
│       └── loader.py         # FAQ data loader
├── frontend/                 # React + Vite UI
│   ├── src/
│   └── vite.config.js
└── .env                      # Environment variables
```

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- Google Gemini API key ([get one here](https://aistudio.google.com/apikey))

### Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_api_key_here
```

Start the backend:

```bash
python app.py
```

The server runs at `http://localhost:5000`. On first start, the `all-MiniLM-L6-v2` model (~80MB) will be downloaded automatically.

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The frontend runs at `http://localhost:5173`.

## 📡 API

### `POST /api/chat`

```json
{ "message": "tell me about kalasalingam college" }
```

**Response:**

```json
{
  "answer": "Kalasalingam Academy of Research and Education (KARE) is...",
  "retrieved_entries": [
    { "question": "...", "category": "...", "score": 0.72 }
  ],
  "model_used": "gemini-2.0-flash",
  "success": true
}
```

### `GET /api/health`

Returns server status, FAQ count, categories, and Gemini availability.

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite |
| Backend | Flask, Flask-CORS |
| Search | sentence-transformers (`all-MiniLM-L6-v2`) |
| Generation | Google Gemini API (`gemini-2.0-flash`) |
| Data | JSON FAQ dataset |

## 📄 License

This project is for educational purposes.
