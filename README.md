# QPSA — Quantization-Probing Self-Adapting Attack

Master's Thesis · Aalborg University 2026

Evaluates whether post-training quantization (FP16 → INT8 → INT4) degrades LLM safety alignment across multiple open-source models, using an adaptive red-teaming attack and LlamaGuard3 as judge.

## Models

**Target Models (tested at FP16 / INT8 / INT4)**
| Model | Size | Organization |
|-------|------|--------------|
| Llama-3.1-8B-Instruct | 8B | Meta |
| Mistral-7B-Instruct-v0.3 | 7B | Mistral AI |
| Gemma-2-9B-it | 9B | Google |
| Phi-3.5-mini-instruct | 3.8B | Microsoft |

**Fixed Models**
| Role | Model | Precision |
|------|-------|-----------|
| Attacker | Qwen2.5-14B-Instruct | INT8 |
| Judge | LlamaGuard3-8B | BF16 |

## Setup
```bash
conda create -n llm_safety python=3.11 -y
conda activate llm_safety
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

```

## Run
```bash
# Example: attack Llama-3.1 at INT4
python3 run_step1.py --precision int4 --model llama
```

## Experiment Matrix
4 target models × 3 precisions

## Dataset
[AdvBench](https://huggingface.co/datasets/walledai/AdvBench) — 520 harmful behaviors,

[HarmBench](https://huggingface.co/datasets/walledai/HarmBench) - 100 harmful prompts 

