# QPSA — Quantization-Probing Self-Adapting Attack

Master's Thesis · Aalborg University 2026

Evaluates whether post-training quantization (FP16 → INT8 → INT4) degrades LLM safety alignment across multiple open-source models, using an adaptive red-teaming attack, LlamaGuard3 as judge, and a Vector DB adaptive defense.

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

## Quantization

Quantization is performed using **OsciQuant PTQ** (Wenshøj et al. 2025, TMLR) — a research-grade post-training quantization method using a UniformQuantizer with full layer control.

```python
from osciquant.quantizers import UniformQuantizer
from osciquant.handler    import attach_weight_quantizers, toggle_quantization

attach_weight_quantizers(model, quantizer=UniformQuantizer(bit_width=4), enabled=False)
toggle_quantization(model, enabled=True)
```

## Attack Strategy

| Attempt | Method |
|---------|--------|
| 1–5 | Chain-of-thought mutation — Qwen2.5 reads the refusal, reasons about why, generates targeted bypass |
| 5–7 | Crescendo escalation — multi-turn attack using model's own responses as foothold |

## Adaptive Defense

Successful jailbreaks are stored in a **ChromaDB vector database** with their precision tag (fp16/int8/int4). Future semantically similar prompts are blocked automatically via cosine similarity search.

```
Jailbreak succeeds
    ↓
Embed prompt → store in ChromaDB with precision tag
    ↓
Next similar prompt → similarity check → blocked if sim ≥ 0.85
    ↓
Defense learns over time → ASR decreases
```

## Setup

```bash
conda create -n llm_safety python=3.11 -y
conda activate llm_safety
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
pip install git+https://github.com/saintslab/osc_reg.git
```

## Dataset

[AdvBench](https://arxiv.org/abs/2307.15043) — 500 harmful behaviors (Zou et al. 2023)  
[HarmBench](https://arxiv.org/abs/2402.04249) — 100 harmful behaviors across 6 categories  
Same prompts used across all precision runs (`seed=42`) for fair comparison.
