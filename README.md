# 🚀 AmbedkarGPT - Advanced RAG System with Comprehensive Evaluation

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-green.svg)](https://github.com/langchain-ai/langchain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An enterprise-grade Retrieval Augmented Generation (RAG) system with multi-document support and comprehensive evaluation framework, developed as part of Kalpit Pvt Ltd AI Intern hiring assessment.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Evaluation Results](#evaluation-results)
- [Project Structure](#project-structure)
- [Technical Deep Dive](#technical-deep-dive)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## 🎯 Overview

AmbedkarGPT is a production-ready RAG system that processes and answers questions from **6 speeches by Dr. B.R. Ambedkar**, featuring:

### Phase 1: Functional RAG Prototype ✅
- **Multi-document Q&A system** processing 6 historical speeches
- **Real-time semantic search** using ChromaDB vector store
- **Local LLM inference** with Ollama (Mistral 7B)
- **Source attribution** showing which documents contributed to answers
- **Interactive command-line interface** for exploratory research

### Phase 2: Comprehensive Evaluation Framework ✅
- **8 evaluation metrics** across 3 categories (Retrieval, Answer Quality, Semantic)
- **3 chunking strategies** tested (Small, Medium, Large)
- **25 test questions** with ground truth answers
- **Automated reporting** with statistical analysis
- **Performance comparison** and optimization recommendations

---

## ✨ Key Features

### 🔍 Advanced RAG Capabilities

| Feature | Description |
|---------|-------------|
| **Multi-Document Retrieval** | Cross-document search and synthesis from 6 speeches |
| **Semantic Search** | Vector similarity using sentence-transformers embeddings |
| **Context-Aware Answers** | LLM-generated responses grounded in retrieved context |
| **Source Tracking** | Shows which documents contributed to each answer |
| **Chunk Optimization** | Tests multiple chunking strategies for optimal performance |

### 📊 Comprehensive Evaluation

| Metric Category | Metrics Implemented |
|----------------|---------------------|
| **Retrieval** | Hit Rate, Mean Reciprocal Rank (MRR), Precision@K |
| **Answer Quality** | Answer Relevance, Faithfulness, ROUGE-L |
| **Semantic** | Cosine Similarity, BLEU Score |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface                           │
│                    (Command-Line Interface)                      │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Question Processing                         │
│                    (Query Encoding Layer)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Vector Retrieval                             │
│         ChromaDB + HuggingFace Embeddings (384-dim)             │
│              Retrieves Top-K Relevant Chunks                     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    LLM Generation                                │
│            Ollama (Mistral 7B) - Local Inference                │
│              Generates Grounded Answers                          │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Response Delivery                             │
│        Answer + Source Attribution + Confidence                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- **Python**: 3.8 or higher
- **Ollama**: Local LLM runtime
- **RAM**: Minimum 8GB (16GB recommended)
- **Disk Space**: 5GB free space

### Quick Start

**1. Install Ollama**

Windows: Download from [ollama.com](https://ollama.com/download/windows)

Linux/MacOS:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**2. Pull Mistral Model**
```bash
ollama pull mistral
ollama run mistral "Test"  # Verify
```

**3. Clone Repository**
```bash
git clone https://github.com/santhosharun18/AmbedkarGPT.git
cd AmbedkarGPT
```

**4. Setup Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**5. Install Dependencies**
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('punkt')"
```

---

## 💻 Usage

### Phase 1: Interactive Q&A System

```bash
python main.py
```

**Example:**
```
❓ Your Question: What is Ambedkar's concept of ideal society?

💡 Answer:
----------------------------------------------------------------------
Ambedkar's concept of an ideal society is one based on liberty, 
equality, and fraternity. He envisioned a society free from the 
constraints of tradition and caste...
----------------------------------------------------------------------

📚 Retrieved from: speech1.txt, speech3.txt (4 chunks)
```

### Phase 2: Run Evaluation

```bash
python evaluation.py
```

**Duration:** 60-90 minutes (tests 3 strategies × 25 questions)

### Generate Report

```bash
python results_analysis.py
```

**Output:** `RESULTS_ANALYSIS.md` with detailed analysis

---

## 📊 Evaluation Results

### Performance Summary

| Strategy | Chunks | Hit Rate | MRR | ROUGE-L | Cosine Sim |
|----------|--------|----------|-----|---------|------------|
| **Small** | 26 | 1.000 | 0.903 | 0.330 | 0.579 |
| **Medium** | 13 | 1.000 | 0.907 | 0.348 | 0.577 |
| **Large** | 7 | 1.000 | 0.913 | 0.317 | 0.587 |

### Key Findings

✅ **Perfect Hit Rate (1.000)** - All strategies retrieved relevant documents  
✅ **High MRR (0.90+)** - Excellent ranking quality  
✅ **100% Success Rate** - All questions answered without errors  
✅ **Optimal Strategy: SMALL** - Best balance across metrics  

---

## 📁 Project Structure

```
AmbedkarGPT/
├── README.md                    # This comprehensive guide
├── .gitignore                   # Git exclusions
├── requirements.txt             # Dependencies
│
├── main.py                      # Phase 1: Q&A system
├── evaluation.py                # Phase 2: Evaluation framework
├── results_analysis.py          # Analysis & reporting
│
├── corpus/                      # 6 speech documents
│   ├── speech1.txt
│   ├── speech2.txt
│   ├── speech3.txt
│   ├── speech4.txt
│   ├── speech5.txt
│   └── speech6.txt
│
├── test_dataset.json           # 25 test questions
├── test_results.json           # Evaluation results
└── RESULTS_ANALYSIS.md         # Analysis report
```

---

## 🔬 Technical Deep Dive

### Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Language | Python 3.8+ | Core implementation |
| RAG Framework | LangChain | Pipeline orchestration |
| Vector Store | ChromaDB | Embedding storage |
| LLM | Ollama (Mistral 7B) | Answer generation |
| Embeddings | sentence-transformers | Vectorization |
| Evaluation | RAGAS, ROUGE, NLTK | Metrics |

### Chunking Strategies

| Strategy | Size | Overlap | Chunks | Use Case |
|----------|------|---------|--------|----------|
| Small | 250 | 50 | 25-35 | Fine-grained |
| Medium | 550 | 75 | 15-25 | Balanced |
| Large | 900 | 100 | 10-20 | Context-heavy |

### Evaluation Metrics

**Retrieval:**
- Hit Rate: Did we find relevant docs?
- MRR: How well are they ranked?
- Precision@K: What % of top-K are relevant?

**Answer Quality:**
- Answer Relevance: Question-answer alignment
- Faithfulness: Grounded in context?
- ROUGE-L: Overlap with ground truth

**Semantic:**
- Cosine Similarity: Vector similarity
- BLEU Score: N-gram overlap

---

## ⚠️ Troubleshooting

### Ollama Connection
```
Error: Could not connect to Ollama
```
**Fix:**
```bash
ollama serve  # Start server
ollama list   # Verify
```

### ChromaDB Warnings
```
Failed to send telemetry event...
```
**Fix:** Harmless warnings. To suppress:
```python
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"
```

### Out of Memory
**Fix:**
- Close other apps
- Use smaller chunk size
- Reduce retrieval K value

### Slow Evaluation
**Fix:**
```python
# In evaluation.py, line 158:
self.test_dataset = self.test_dataset[:5]  # Test 5 questions
```

### NLTK Missing
```
LookupError: Resource punkt not found
```
**Fix:**
```python
import nltk
nltk.download('punkt')
```

---

## 🚀 Performance Optimization

### Faster Evaluation
- Reduce test set: `[:10]` instead of all 25
- Enable GPU: `device='cuda'`
- Parallel processing

### Better Accuracy
- Increase K: `search_kwargs={"k": 6}`
- Lower temperature: `temperature=0.1`
- Fine-tune chunk sizes

---

## 📈 Future Enhancements

- [ ] Web UI (Streamlit/Gradio)
- [ ] REST API (FastAPI)
- [ ] Multi-language support
- [ ] Docker deployment
- [ ] Response caching
- [ ] Fine-tuned models

---

## 📄 License

MIT License - Copyright (c) 2025 Santhos Arun

---

## 🙏 Acknowledgments


- **Dr. B.R. Ambedkar** - Speech content
- **LangChain, ChromaDB, Ollama** - Technology stack
- **HuggingFace, RAGAS** - Embeddings & evaluation

---

## 👤 Author

**Santhosh Arun**

- GitHub: https://github.com/santhosharun18
- Email: santhosharun31@gmail.com
- LinkedIn: https://www.linkedin.com/in/santhosh-d-2a6ba3221/
- Portfolio: https://santhosharun18.github.io/my-portfolio/

---

## 📊 Project Status

| Component | Status |
|-----------|--------|
| Phase 1: RAG System | ✅ Complete |
| Phase 2: Evaluation | ✅ Complete |
| Documentation | ✅ Complete |
| Testing | ✅ 100% Success |

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

**Built with ❤️ by Santhos Arun | November 2025**

</div>
