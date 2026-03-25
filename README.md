---
title: Autonomous ICD-10 Coding Tool
emoji: 🏥
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# Autonomous ICD-10 Coding Tool

A production-ready system for extracting clinical entities from medical prescriptions using NLP (scispaCy) and LLMs (Claude), mapping them to standardized ICD-10 codes.

## 🚀 Features
- **OCR & Text Extraction**: Efficiently extract text from prescription images or PDFs.
- **NLP Pipeline**: Specialized medical entity recognition using `scispaCy` and `negspaCy`.
- **LLM Reasoning**: Advanced clinical entity extraction and multi-dimensional analysis via Anthropic Claude API.
- **ICD-10 Mapping**: Resolution of clinical entities to validated ICD-10 codes.
- **Audit System**: Complete tracking of extraction and mapping history.

## 🛠 Tech Stack
- **Backend:** Python FastAPI, SQLAlchemy (SQLite)
- **Frontend:** Streamlit
- **NLP:** scispaCy, negspaCy
- **LLM:** Anthropic Claude API (Claude 3.5 Sonnet)
- **Infrastructure:** Docker, Docker Compose

## 📁 Project Structure
```text
medical-icd-mapper/
├── backend/            # FastAPI Backend
│   ├── api/            # API Routes
│   ├── core/           # Config, Logging, Security
│   ├── database/       # SQLAlchemy Session & Engine
│   ├── services/       # NLP, LLM, ICD services
│   ├── schemas/        # Pydantic models
│   └── tests/          # Unit & Integration tests
├── frontend/           # Streamlit Frontend
├── database/           # SQLite DB, Migrations & Seed scripts
├── config/             # Environment templates
├── docker/             # Docker configuration
├── docs/               # Documentation
└── scripts/            # Automation scripts
```

## ⚙️ Setup Instructions

### Prerequisites
- Python 3.10+
- Anthropic API Key

### Local Development
1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd medical-icd-mapper
   ```

2. **Backend Setup:**
   ```bash
   cd backend
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Frontend Setup:**
   ```bash
   cd ../frontend
   pip install -r requirements.txt
   ```

4. **Environment Variables:**
   Copy `config/.env.example` to `config/.env` and fill in your credentials.

5. **Run with Docker:**
   ```bash
   docker-compose up --build
   ```

## 🏗 Architecture Overview
The system follows a service-oriented architecture where the FastAPI backend handles intensive processing (NLP/LLM/DB) and exposes a RESTful API. The Streamlit frontend provides a lightweight, interactive interface for healthcare professionals to upload documents and review results.

## 🧪 Testing
Run backend tests:
```bash
cd backend
pytest
```

## 📄 License
MIT License
