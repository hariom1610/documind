# Documind AI
# 📄 Document Analysis API

An intelligent, fully open-source document processing REST API that automatically extracts, analyses, and summarises content from **PDF**, **DOCX**, and **image** files using state-of-the-art local AI. Built for the Track 2: AI-Powered Document Analysis & Extraction hackathon challenge.

---

## Description

This API accepts documents as Base64-encoded strings (or via a convenient testing script) and returns structured JSON containing:
- **AI-generated summary** of the document content
- **Named entity extraction**: person names, dates, organizations, monetary amounts, and locations
- **Sentiment classification**: Positive, Neutral, or Negative

The system uses a completely local, multi-stage architecture: format-specific text extraction → localized Hugging Face Transformer models (NER, Sentiment, Summarization) + targeted Regex → structured response. **No external APIs are required.**

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Framework** | FastAPI + Uvicorn |
| **PDF Extraction** | PyMuPDF (fitz) — preserves layout; falls back to OCR for scanned PDFs |
| **DOCX Extraction** | python-docx — extracts paragraphs + tables |
| **OCR (Images)** | pytesseract + Pillow (Tesseract 4 LSTM engine) |
| **Local AI Models** | Hugging Face `transformers` & `torch` (BART, DistilBERT, BERT-NER) |
| **Safety Fallbacks** | Pure Python TF-IDF extractive summarization algorithm & Regex patterns |
| **Auth** | Custom x-api-key header middleware |
| **Deployment** | Docker (Python 3.11 slim + Tesseract) |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/hariom1610/documind-ai.git
cd documind-ai
```

### 2. Install system dependencies (Tesseract OCR)
*(Note: If you plan to deploy via Docker, Tesseract is pre-configured and installs automatically).*
```bash
# Ubuntu / Debian
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# macOS
brew install tesseract
```

### 3. Install Python dependencies
```bash
python -m venv venv
source venv/bin/activate        # Windows: .\venv\Scripts\activate
pip install -r requirements.txt
```

### 4. Set environment variables
Create a `.env` file in the root directory and invent your own API Key to secure the server:
```bash
cp .env.example .env
# Edit .env and securely invent your API_KEY (e.g. API_KEY=hackathon_secret_123)
```

### 5. Run the application
```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`  
Interactive docs at `http://localhost:8000/docs`

---

## Testing Tools

Tired of using Postman or Swagger to copy-paste giant Base64 strings? Use the included local test script!
While the server is running, open a new terminal:
```bash
python test_api.py "path_to_your_sample_document.pdf"
```

---

## Docker Deployment

The Dockerfile is fully configured for hackathon evaluations.
```bash
# Build and run
docker build -t documind-ai .
docker run -p 8000:8000 --env-file .env documind-ai
```

---

## API Usage

### Endpoint
```
POST /api/document-analyze
```

### Headers
| Header | Value |
|---|---|
| `Content-Type` | `application/json` |
| `x-api-key` | Your secret API key |

### Request Body
```json
{
  "fileName": "sample.pdf",
  "fileType": "pdf",
  "fileBase64": "<base64-encoded file content>"
}
```

`fileType` must be one of: `pdf`, `docx`, `image`

### Success Response (200)
```json
{
  "status": "success",
  "fileName": "invoice.pdf",
  "summary": "This document is an invoice issued by ABC Pvt Ltd to Ravi Kumar on 10 March 2026 for a total amount of ₹10,000 for consulting services.",
  "entities": {
    "names": ["Ravi Kumar"],
    "dates": ["10 March 2026"],
    "organizations": ["ABC Pvt Ltd"],
    "amounts": ["₹10,000"],
    "locations": []
  },
  "sentiment": "Neutral"
}
```

---

## Approach

### Text Extraction Strategy

**PDF files** are handled by PyMuPDF (`fitz`), which preserves document layout and extracts text page by page. If extracted text is sparse (< 50 characters — indicating a scanned/image-based PDF), the system automatically falls back to rendering each page as a high-resolution image and running Tesseract OCR on it.

**DOCX files** are processed with `python-docx`, which iterates through all paragraphs and table cells to ensure complete text coverage.

**Image files** are processed via `pytesseract` with image preprocessing configured for optimal accuracy.

### Local AI Analysis Strategy (No External APIs)

The application handles text analysis securely inside the host environment using a pipeline of modular, specialized Hugging Face transformer models:

1. **Named Entity Recognition (NER)**: Uses `dslim/bert-base-NER` to extract standard entities (Names, Organizations, Locations) while sweeping the text in smart chunks to manage memory efficiency safely under 512 tokens.
2. **Sentiment Analysis**: Uses `distilbert-base-uncased-finetuned-sst-2-english` to accurately calculate the prevailing tone (Positive, Negative, Neutral).
3. **Structured Regex Fallbacks**: Since classic NER systems struggle natively with shifting currency formats and date symbols, a highly optimized Python Regex parser securely lifts absolute `Amounts` and `Dates` from the text.
4. **Resilient Summarization**: Utilizes the `distilbart-cnn-12-6` architecture. If PyTorch encounters backend loading errors on the host machine, the application triggers a **Pure-Python Extractive Mathematical Summarizer** backup algorithm ensuring an accurate summary is returned 100% of the time without crashing the server.

---

## Health Check

```bash
curl http://localhost:8000/health
```

```json
{
  "status": "healthy",
  "api_key_configured": true
}
```