# QPSA — Quantization-Probing Self-Adapting Attack

**Master's Thesis · Aalborg University 2026**

This repository contains the complete code, configuration files, and experimental results for the thesis *"QPSA: Quantization-Probing Self-Adapting Attack"*. The project systematically evaluates how post‑training quantization (PTQ) affects the safety alignment of large language models under adaptive chain‑of‑thought attacks, and assesses a hybrid stateful defence that combines a persistent vector store with an online classifier.

---

## 📌 Overview

- **Quantization**: Half‑Quadratic Quantization (HQQ) applied on‑the‑fly to four instruction‑tuned models (Llama‑3.1‑8B, Mistral‑7B, Gemma‑2‑9B, Phi‑3.5‑mini) at FP16, INT8, and INT4.
- **Adaptive Attacker**: Qwen2.5‑7B‑Instruct that iteratively mutates harmful prompts using chain‑of‑thought reasoning (max 5 attempts per goal). Temperature decays from 1.0 to 0.35.
- **Safety Judge**: Qwen3Guard‑Gen‑8B providing Safe/Unsafe/Controversial labels and a refusal flag (Yes/No). (Early experiments used LlamaGuard‑3 for comparison.)
- **Hybrid Defence**: ChromaDB vector store + online SGD classifier; a prompt is blocked if risk score `R = α·s_max + (1-α)·c` exceeds threshold τ (α=0.6, τ=0.75).
- **Evaluation**: Attack Success Rate (ASR), controversial rate, average attempts per success

---

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/your-username/adaptive_red_teaming.git
cd adaptive_red_teaming

conda env create -f environment.yml
conda activate llm_safety

pip install -r requirements.txt
pip install git+https://github.com/mobiusml/hqq.git
