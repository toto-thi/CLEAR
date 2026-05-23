# CLEAR: Critique-Led Explainable Agents with Image Retrieval for Grounded Skin-Lesion Diagnosis

This repository implements an AI-driven dermatological diagnosis pipeline using a multi-agent system orchestrated with **LangGraph**. Developed as part of a Master's Thesis, it mimics a clinical workflow to analyze skin lesion images, leveraging Large Language Models (LLMs) and a **Visual Knowledge Base (VKB)** for grounded, evidence-based reasoning.

## Overview

- **Multi-Agent Orchestration:** Specialized agents (Lab, Triage, Diagnosis, Critique, Synthesizer) collaborate via LangGraph to provide structured clinical insights.
- **Visual Knowledge Base (VKB):** Uses **BiomedCLIP** embeddings stored in **Qdrant** to retrieve similar reference cases for few-shot diagnostic prompting.
- **Grounded Reasoning:** Agents are instructed to follow a **Hierarchy of Reasoning** (Global Pattern > High-Specificity Clues > Ambiguous Cues) and perform an **Invalidation Check**.
- **Comprehensive Evaluation:** Built-in metrics for Rationale Specificity, Groundedness, and diagnostic accuracy.

## 🏗 System Architecture

The workflow mimics a professional dermatological consultation:

1.  **Lab Technician:** Extracts objective dermoscopic features (pigment networks, vascular structures, etc.) from the image.
2.  **Triage Agent:** Classifies the lesion into a disease family (Melanocytic, Keratinocytic, or Fibrohistiocytic).
3.  **Diagnosis Agent:** Proposes primary and differential diagnoses by synthesizing lab findings and reference cases retrieved from the VKB.
4.  **Critique Agent:** Acts as a clinical reviewer, challenging the initial proposal and constructing a strong alternative hypothesis.
5.  **Synthesizer (Chief Medical Officer):** Weighs the initial diagnosis against the critique, applies clinical principles, and provides a final verdict and management recommendation (e.g., biopsy).

## 🚀 Getting Started

### Prerequisites

- **Docker & Docker Compose:** Required to run the Qdrant vector database.
- **Python 3.12+:** Managed via [`uv`](https://github.com/astral-sh/uv).
- **API Keys:** Access to Google Gemini and/or OpenAI GPT-5.

### Installation

1.  **Download/Clone the repository**

2.  **Infrastructure Setup:**
    Start the Qdrant database using Docker Compose:
    ```bash
    docker-compose up -d
    ```

3.  **Python Environment:**
    Install dependencies using `uv`:
    ```bash
    uv sync
    ```

4.  **Environment Variables:**
    Copy `example.env` to `.env` and provide your API keys:
    ```bash
    cp example.env .env
    ```

### Knowledge Base Initialization

Before running the diagnosis pipeline, you must index the dermatological dataset:
1.  Ensure your dataset is placed in the `dataset/` directory.
2.  Run the `experiments/create_kb.ipynb` notebook to generate embeddings and populate the Qdrant collection.

## 📊 Usage

### Running the Pipeline

You can execute the full diagnostic workflow on the test dataset using the provided CLI:

```bash
# Run with Google Gemini
python experiments/final-pipeline.py --provider gemini --output final-gemini.jsonl

# Run with OpenAI GPT-5
python experiments/final-pipeline.py --provider openai --output final-gpt.jsonl
```

### Evaluation

Use the notebooks in the `evaluation/` directory to analyze results:
- `score_calculator.ipynb`: Primary notebook for calculating accuracy, specificity, and groundedness metrics.
- `ablation_score.ipynb`: Used to measure the impact of specific agents or logic changes.

## 📁 Repository Structure

- `agents/`: Markdown prompt templates and JSON schemas for each agent.
- `dataset/`: Storage for dermatological images and metadata.
- `eval_utils/`: Core Python modules for evaluation metrics and data processing.
- `evaluation/`: Jupyter notebooks for scoring and analysis.
- `experiments/`: Implementation of the LangGraph workflow, LLM setup, and VKB retrieval.
- `results/`: Output directory for `.jsonl` result files and performance plots.

---
*This research explores the intersection of Agentic AI and clinical decision support in dermatology.*
