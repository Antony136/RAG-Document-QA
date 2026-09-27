# RAG Document Q&A

A local Retrieval-Augmented Generation (RAG) application for asking questions about uploaded PDF documents.

The system extracts and chunks PDF text, generates embeddings, stores them in PostgreSQL with `pgvector`, retrieves and reranks relevant chunks, and uses a local Qwen 2.5 7B model through Ollama to generate grounded answers with source information.

The project was built as a practical learning project to understand how modern RAG systems are designed, evaluated, tested, and improved.

---

## Architecture

```text
                         DOCUMENT INGESTION

PDF
 ↓
Text Extraction
 ↓
Sentence-based Chunking
 ↓
Overlapping Chunks
 ↓
Embedding Generation
 ↓
PostgreSQL + pgvector


                         QUESTION ANSWERING

User Question
 ↓
Query Normalization
 ↓
Conversation-aware Query Rewrite
 ↓
Query Correction
 ↓
Multi-query Generation
 ↓
Vector Search
 ↓
Reciprocal Rank Fusion (RRF)
 ↓
Cross-encoder Reranking
 ↓
Rerank Threshold
 ↓
Context Construction
 ↓
Qwen 2.5 7B
 ↓
Grounded Answer
 ↓
Sources + Page Information
```

---

## Features

### Document Ingestion

* PDF document upload
* PDF text extraction using `pypdf`
* Sentence-based text chunking
* Overlapping chunks
* Local embedding generation
* PostgreSQL document storage
* PostgreSQL `pgvector` storage

### Retrieval

* Semantic vector search
* Document-specific retrieval
* Query normalization
* Conversation-aware query rewriting
* Query spelling correction
* Multi-query retrieval
* Reciprocal Rank Fusion (RRF)
* Cross-encoder reranking
* Reranking confidence threshold
* Source and page metadata

### Generation

* Local LLM inference using Ollama
* Qwen 2.5 Coder 7B
* Context-grounded answer generation
* Refusal when relevant information cannot be found
* Conversation-aware follow-up questions
* Source-aware answers

### Evaluation

* Golden evaluation dataset
* Recall@K
* Precision@K
* Hit Rate@K
* Mean Reciprocal Rank (MRR)
* nDCG@K
* Answer correctness
* Answer relevance
* Faithfulness
* Context relevance
* Retrieval latency
* Generation latency
* End-to-end latency
* Automated evaluation scripts
* Automated evaluation tests

### Testing

* Application unit/integration tests
* API tests
* Retrieval tests
* Generation tests
* Evaluation metric tests
* LLM judge validation tests

---

## Tech Stack

### Backend

* Python
* FastAPI
* PostgreSQL
* pgvector
* psycopg
* pypdf
* Sentence Transformers

### LLM

**Qwen 2.5 Coder 7B**

Running locally through:

* Ollama

### Embedding Model

**nomic-embed-text**

### Reranker

**cross-encoder/ms-marco-MiniLM-L-6-v2**

### Frontend

* React
* Vite

---

## Project Structure

```text
rag-document-qa/
│
├── app/
│   ├── core/
│   │   ├── config.py
│   │   ├── database.py
│   │   └── test_database.py
│   │
│   ├── ingestion/
│   │   ├── extract.py
│   │   ├── chunker.py
│   │   ├── embeddings.py
│   │   └── pipeline.py
│   │
│   ├── retrieval/
│   │   ├── vector_store.py
│   │   ├── similarity.py
│   │   ├── search.py
│   │   ├── reranker.py
│   │   ├── query_transform.py
│   │   ├── multi_query.py
│   │   ├── fusion.py
│   │   ├── query_correction.py
│   │   ├── vocabulary.py
│   │   ├── query_rewrite.py
│   │   ├── query_normalization.py
│   │   └── query_rewriter.py
│   │
│   ├── generation/
│   │   ├── llm.py
│   │   ├── prompts.py
│   │   └── sources.py
│   │
│   ├── storage/
│   │   ├── chunks.py
│   │   └── documents.py
│   │
│   ├── evaluation/
│   │   └── grounding.py
│   │
│   ├── api/
│   │   └── documents.py
│   │
│   ├── rag.py
│   └── main.py
│
├── evaluation/
│   ├── dataset/
│   │   └── questions.json
│   │
│   ├── metrics/
│   │   └── retrieval_metrics.py
│   │
│   ├── generation/
│   │   └── llm_judge.py
│   │
│   ├── tests/
│   │   ├── test_retrieval_metrics.py
│   │   └── test_llm_judge.py
│   │
│   ├── results/
│   │
│   ├── validate_dataset.py
│   ├── run_evaluation.py
│   ├── run_generation_evaluation.py
│   └── run_all.py
│
├── frontend/
├── documents/
├── experiments/
├── tests/
│
├── .env
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

---

# Setup

## 1. Clone the Repository

```bash
git clone <your-repository-url>
cd rag-document-qa
```

## 2. Create a Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.venv\Scripts\Activate.ps1
```

## 3. Install Dependencies

```powershell
pip install -r requirements.txt
```

---

## 4. Install and Start Ollama

Install Ollama and make sure it is running.

Pull the required models:

```powershell
ollama pull qwen2.5-coder:7b
ollama pull nomic-embed-text
```

The reranker model is loaded through Sentence Transformers when required.

---

## 5. Configure PostgreSQL

Create a PostgreSQL database:

```sql
CREATE DATABASE rag_document_qa;
```

Connect to the database and enable `pgvector`:

```sql
CREATE EXTENSION vector;
```

Create the required tables according to the schema used by the application.

The database stores:

* Documents
* Document chunks
* Embeddings
* Chunk metadata

---

## 6. Configure Environment Variables

Create a `.env` file:

```env
DATABASE_URL=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_document_qa

DATABASE_URL_TEST=postgresql://postgres:YOUR_PASSWORD@localhost:5432/rag_document_qa_test
```

> Do not commit `.env` to GitHub.

---

# Running the Application

## Backend

From the project root:

```powershell
uvicorn app.main:app --reload
```

The backend normally runs at:

```text
http://127.0.0.1:8000
```

---

## Frontend

Open another terminal:

```powershell
cd frontend

npm install
npm run dev
```

Open the Vite development URL shown in the terminal.

---

# RAG Pipeline

The application follows a complete RAG pipeline.

### 1. Document Ingestion

A PDF is uploaded and its text is extracted.

### 2. Chunking

The extracted text is divided into smaller overlapping sentence-based chunks.

### 3. Embeddings

Each chunk is converted into a vector representation using:

```text
nomic-embed-text
```

### 4. Vector Storage

Embeddings and metadata are stored in PostgreSQL using `pgvector`.

### 5. Query Processing

Before retrieval, the user's question can go through:

```text
Normalization
      ↓
Conversation-aware rewriting
      ↓
Spelling correction
```

### 6. Multi-query Retrieval

Multiple search queries are generated from the processed question to improve retrieval coverage.

### 7. Vector Search

Each query searches the PostgreSQL vector database for semantically similar chunks.

### 8. Reciprocal Rank Fusion

Results from the different generated queries are combined using Reciprocal Rank Fusion (RRF).

### 9. Cross-encoder Reranking

The retrieved candidates are reranked using:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### 10. Confidence Threshold

Low-confidence retrieval results can be rejected using the reranking threshold.

### 11. Context Construction

The highest-ranked chunks are formatted into the context supplied to the LLM.

### 12. Answer Generation

Qwen 2.5 Coder 7B generates an answer using the retrieved context.

The model is instructed to avoid using information outside the supplied document context.

---

# Testing

The project separates **application testing** from **evaluation-code testing**.

## Application Tests

Run:

```powershell
pytest tests -q
```

The current application test suite contains:

```text
115 passed
2 dependency warnings
```

The warnings are dependency deprecation warnings and do not currently cause test failures.

---

## Evaluation Tests

Evaluation code has its own tests:

```powershell
pytest evaluation/tests -q
```

The current evaluation test suite contains:

```text
18 passed
```

These tests verify the correctness of evaluation components such as:

* Recall
* Precision
* Hit Rate
* MRR
* nDCG
* LLM judge validation

---

# RAG Evaluation

The project contains a dedicated evaluation pipeline rather than relying only on manually checking answers.

A golden dataset containing **40 questions** was created from the documents currently stored in the database.

The dataset contains questions covering:

* MLOps
* Generative AI / RAG
* Social Network Engineering
* CSE curriculum material

Each question contains:

* Question
* Expected answer
* Relevant document chunks
* Document ID
* Difficulty
* Category

---

## Retrieval Metrics

The retrieval evaluator measures:

* Recall@1
* Recall@3
* Recall@5
* Recall@10
* Precision@1
* Precision@3
* Precision@5
* Precision@10
* Hit Rate@1
* Hit Rate@3
* Hit Rate@5
* Hit Rate@10
* nDCG@1
* nDCG@3
* nDCG@5
* nDCG@10
* MRR

These metrics measure different aspects of retrieval quality, including whether relevant chunks are retrieved and how highly they are ranked.

---

## Generation Metrics

Generated answers are additionally evaluated using an LLM judge.

The current evaluation measures:

* Answer correctness
* Answer relevance
* Faithfulness
* Context relevance

Each dimension is scored from **1 to 5**.

The generation evaluation uses the same local Qwen model family for judging, so these scores should be interpreted as an **internal evaluation signal**, not as an objective human benchmark.

---

# Latest Evaluation Results

The latest complete evaluation was performed on:

```text
40 questions
951 document chunks
```

## Retrieval

| Metric       |  Score |
| ------------ | -----: |
| Recall@1     | 0.3267 |
| Recall@3     | 0.5713 |
| Recall@5     | 0.7688 |
| Recall@10    | 0.7800 |
| Precision@1  | 0.5500 |
| Precision@3  | 0.3417 |
| Precision@5  | 0.2750 |
| Precision@10 | 0.2265 |
| Hit Rate@1   | 0.5500 |
| Hit Rate@3   | 0.7750 |
| Hit Rate@5   | 0.9000 |
| Hit Rate@10  | 0.9250 |
| nDCG@1       | 0.5500 |
| nDCG@3       | 0.5612 |
| nDCG@5       | 0.6397 |
| nDCG@10      | 0.6460 |
| MRR          | 0.6823 |

For example, a Hit Rate@5 of `0.9000` means that **36 of the 40 evaluated questions had at least one expected relevant chunk within the top 5 retrieved results**.

---

## Generation

| Metric            |    Score |
| ----------------- | -------: |
| Correctness       | 4.75 / 5 |
| Relevance         | 4.92 / 5 |
| Faithfulness      | 4.70 / 5 |
| Context Relevance | 4.85 / 5 |

These scores are generated by the project's local LLM judge and are intended as an internal engineering signal.

---

## Latency

| Stage                              |     Average |
| ---------------------------------- | ----------: |
| Retrieval                          |  3537.70 ms |
| Generation                         |  7844.04 ms |
| LLM Judge                          | 10939.26 ms |
| Total evaluation time per question | 22321.00 ms |

The LLM judge is substantially slower than normal answer generation because every evaluated answer requires an additional model call.

---

# Running Evaluation

## Validate the Golden Dataset

```powershell
python -m evaluation.validate_dataset
```

---

## Retrieval Evaluation

```powershell
python -m evaluation.run_evaluation
```

---

## Generation Evaluation

```powershell
python -m evaluation.run_generation_evaluation
```

A smaller evaluation can be run during development:

```powershell
python -m evaluation.run_generation_evaluation --limit 5
```

---

## Complete Project Verification

The project provides a single command that runs the complete verification pipeline:

```powershell
python -m evaluation.run_all
```

It performs:

```text
Dataset Validation
       ↓
Retrieval Evaluation
       ↓
Generation Evaluation
       ↓
Application Tests
       ↓
Evaluation Tests
```

This makes it possible to verify the project from one command after making changes.

---

# Evaluation Limitations

The evaluation results should not be interpreted as production-grade benchmarking.

The current evaluation has several limitations:

* The golden dataset contains 40 questions.
* Relevant chunks were manually annotated.
* Generation quality is evaluated using an LLM judge.
* The generator and judge use the same local Qwen model family.
* Latency depends heavily on the local hardware and model configuration.
* The evaluation corpus represents the documents currently loaded into the project database.
* Retrieval metrics measure performance against the current manually annotated dataset rather than a universal RAG benchmark.

The purpose of this evaluation is to establish a reproducible internal baseline and provide measurable feedback while developing the RAG system.

---

# Learning Goals

This project was built to gain practical experience with:

* Retrieval-Augmented Generation
* Document ingestion
* Text chunking
* Embeddings
* Vector databases
* PostgreSQL
* pgvector
* Semantic search
* Query normalization
* Query rewriting
* Query correction
* Multi-query retrieval
* Reciprocal Rank Fusion
* Cross-encoder reranking
* Grounded generation
* Local LLM inference
* FastAPI
* React
* Automated testing
* Retrieval evaluation
* Generation evaluation
* LLM-as-a-judge evaluation
* Latency measurement

---

# Project Status

**Project 2 — RAG Document Q&A is functionally complete and evaluated.**

The complete application flow has been verified:

```text
PDF Upload
    ↓
Text Extraction
    ↓
Chunking
    ↓
Embedding
    ↓
Vector Storage
    ↓
Query Processing
    ↓
Multi-query Retrieval
    ↓
RRF
    ↓
Cross-encoder Reranking
    ↓
Context Construction
    ↓
LLM Generation
    ↓
Grounded Answer
    ↓
Source Display
```

The project now has:

* Complete RAG pipeline
* Local LLM inference
* Multi-stage retrieval
* Reranking
* Conversation-aware querying
* Automated application tests
* Dedicated evaluation tests
* Golden evaluation dataset
* Retrieval metrics
* Generation quality evaluation
* Latency measurement
* One-command evaluation

---

# Next Project

This project is part of a progressive AI engineering learning path:

```text
Project 1
LLM Playground
      ↓
Project 2
RAG Document Q&A
      ↓
Project 3
Agents + Tool Calling
      ↓
Project 4
AI Full-Stack Application
      ↓
Project 5
Advanced AI System
```

The goal of this progression is to move from understanding individual LLM capabilities to building complete AI systems involving retrieval, tools, agents, APIs, evaluation, and production-oriented architecture.
