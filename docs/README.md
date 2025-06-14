# 🛡️ AI-Powered Certificate Forgery Detection

A modular, locally deployable certificate forgery detection system that leverages **Vision Transformers**, **OCR + Metadata analysis**, **AI text detection**, and **LLM-based reasoning** to evaluate whether certificates are genuine or fraudulent. Built using **FastAPI**, **n8n**, **MongoDB**, and **Ollama**.

![Workflow Diagram](forgery-workflow.png)

---

## 🚀 Key Features

* 🔍 **Multi-signal Detection**: Combines ViT classifier, AI-text detector, OCR anomaly analysis, and metadata insights
* 🤖 **Explainable Classification**: LLM-generated reasoning for every decision
* 🧠 **Ollama LLM and HuggingFace Classifiers**:  LLMs for text formatting and fraud reasoning, classifiers to detect ai generated text and forged documents 
* 🔗 **FastAPI Server**: Central backend exposing `/analyze_image` and `/classify_text` endpoints
* 📅 **n8n Workflow**: Fully visual, file-driven pipeline with form input, HTTP orchestration, and MongoDB storage
* 📊 **MongoDB Storage**: Final results stored with filename, classification, confidence, and LLM explanation

---

## 🌐 API Endpoints (FastAPI Server)

| Endpoint         | Description                                                                      |
| ---------------- | -------------------------------------------------------------------------------- |
| `/analyze_image` | Receives a certificate image, returns OCR text, metadata, and ViT classification |
| `/classify_text` | Classifies cleaned OCR text as Human- or AI-written                              |

---

## 📂 Repository Structure

```
document_forgery_detection/
├── data/
│   ├── train/ valid/ test/                 # Image dataset (Roboflow-style) for training VIT
│── tests/                                  # Image dataset for testing n8n workflow
├── scripts/
│   ├── train_vit_classifier.py             # Fine-tune ViT on certificate images
│   ├── test_pred_vit_classifier.py         # Evaluate on test set or predict on new image
│   ├── extract_metadata.py                 # Extract EXIF, PIL, hachoir, XMP
│   ├── ocr_plus_analysis.py                # HOCR OCR + anomaly detection
│   ├── classify_text_humanOai.py           # Use RoBERTa-based AI-text detector
│   └── server.py                           # FastAPI server with endpoints
├── workflows/
│   └── Document_Forgery_Detection_workflow.json  # n8n flow for full pipeline
├── docs/
│   ├── README.md
│   └── technical-approach.md
├── .env
├── compose.yaml
└── pyproject.toml
```

---

## 🧪 Run Locally

### Prerequisites

* Python 3.10+
* Docker & Docker Compose
* Tesseract OCR (ensure it's in `PATH`)
* Ollama installed locally with `llama3` and `qwen3` models pulled

### Setup (Python & FastAPI)

```bash
uv venv
uv sync
uv run scripts/server.py
```

API server runs at: `http://localhost:8000`

---

## 🔄 n8n Workflow

### Launch Instructions

1. Start n8n:

   ```bash
   docker compose up n8n
   ```

2. Open Editor UI:
   [http://localhost:5678](http://localhost:5678)

3. Trigger Form:
   [http://localhost:5678/form-test/2e96ea7a-c5a1-44cf-a41e-3b7c20cc9830](http://localhost:5678/form-test/2e96ea7a-c5a1-44cf-a41e-3b7c20cc9830)

### Workflow Description

* 📄 **Form Trigger**: Accepts `.jpg`/`.png` certificate files.
* 🥈 **Binary Split + POST**: Sends each file to `/analyze_image`.
* 🧠 **LLM Agent #1**: Formats OCR text (Llama3).
* 🤖 **AI Text Detection**: Classifies as Human- or AI-generated.
* 📊 **LLM Agent #2**: Makes final decision based on all reports.
* 🗏️ **MongoDB Insert**: Stores to `Document_Store` in `{ filename, output }` format.

---

## 🧠 Example Output

```json
{
  "filename": "sample.jpg",
  "output": {
    "document_classification": "fraudulent",
    "confidence": 95.5,
    "explanation": "The VIT_classifier_report labeled the document as fraudulent with 94.2% confidence, and the AI_detection identified it as AI-generated with 96.8% confidence. These findings strongly indicate the document is synthetic, with consistent signals across both analyses."
  }
}
```

---

## 📌 Future Enhancements

* Fine-tune the AI text detector for domain-specific certificates
* Add PDF support and multipage OCR handling
* Stream responses in real time via WebSocket
* Build an admin dashboard UI for visualization

---

## 📜 License

MIT License
# 📄 technical-approach.md

## 1. Overview

The **AI-Powered Certificate Authentication System** is a multimodal document forgery detection framework. It combines a fine-tuned Vision Transformer (ViT), OCR-based analysis, and AI-generated text detection via RoBERTa, all orchestrated through a locally hosted **n8n** workflow and served via **FastAPI**.

---

## 2. System Architecture

### Components

* **ViT Classifier**: Fine-tuned `vit-base-patch16-224-in21k` for certificate classification.
* **OCR + Visual Forensics**: Detect spacing/vertical inconsistencies, font anomalies.
* **AI Text Detection**: Uses `roberta-base-openai-detector` to flag AI-generated content.
* **Metadata Analysis**: Extracts and analyzes EXIF, quantization, and tool signature data.
* **n8n Orchestration**: Connects all steps into a structured verification pipeline.
* **MongoDB**: Stores results with session tracking and filename.

---

## 3. Document Flow Pipeline

```mermaid
graph TD
A[Form Trigger] --> B[Split Upload List]
B --> C[Loop Over Images]
C --> D1[HTTP: /analyze_image]
C --> D2[HTTP: /classify_text]
D1 --> E[Combine With Classify Text Result]
E --> F[LLM JSON Formatter (Ollama)]
F --> G[Save to MongoDB]
```

* **D1** uses `/analyze_image`: returns metadata, OCR analysis, and ViT result.
* **D2** uses `/classify_text`: returns AI detection result.
* **F** formats all outputs into a final JSON summary.

---

## 4. Machine Learning Models

### ViT Classifier

* Dataset: Roboflow-based certificate images (345 samples, 2-class: `fake`, `true`)
* Architecture: `vit-base-patch16-224-in21k`
* Metrics: F1 = 0.98, Accuracy = 0.98 on test set
* Inference output: `{'label': 'fraudulent' | 'unedited', 'confidence': float}`

### AI Text Detection

* Model: `openai-community/roberta-base-openai-detector`
* Input: OCR'd text from scanned certificate
* Output: `{'label': 'AI-generated' | 'Human-written', 'confidence': float}`

---

## 5. OCR + Analysis

* Extracted using Tesseract HOCR format
* Features analyzed:

  * **Spacing Gaps** (>30px between word bounding boxes)
  * **Vertical Misalignment** (baseline shifts >10px)
  * **Font Anomalies** (inconsistent font family across words)

---

## 6. API Endpoints

### `POST /analyze_image`

* Input: Multipart/form-data or raw image
* Output:

```json
{
  "metadata_report": { ... },
  "ocr_analysis_report": { ... },
  "VIT_classifier_report": { ... },
  "extracted_text": "..."
}
```

### `POST /classify_text`

* Input:

```json
{ "text": "OCR result string" }
```

* Output:

```json
{
  "AI_detection": {
    "label": "AI-generated",
    "confidence": 0.9842
  }
}
```

---

## 7. Database Schema

MongoDB Collection: `certificate_analysis`

Each document:

```json
{
  "filename": "xyz.jpg",
  "output": {
    "document_classification": "fraudulent",
    "confidence": 97.3,
    "explanation": "Based on high-confidence fraud label from ViT model and AI-generated text flag."
  }
}
```

---

## 8. Deployment & Usage

* **Run Server**: `uvicorn main:app --reload --port 8000`
* **Form Link (trigger)**: `http://localhost:5678/form-test/<id>` (use `host.docker.internal` inside Docker)
* **Final Storage**: Result JSONs saved to MongoDB and viewable in Mongo Express.

---

## 9. Evaluation & Metrics

| Task                  | Model                 | Accuracy | Confidence Avg |
| --------------------- | --------------------- | -------- | -------------- |
| Image Classification  | ViT (fine-tuned)      | 93.2%    | 0.90           |
| AI Text Detection     | RoBERTa-base-detector | \~90.4%  | 0.93           |
| OCR Anomaly Detection | Heuristic thresholds  | -        | -              |

---

## 10. Future Work

* Add ELA (Error Level Analysis) for JPEG tampering
* Extend to multilingual certificates
* Implement blockchain certificate hash comparison
* Streamline results to include visual heatmaps for fraud indicators
