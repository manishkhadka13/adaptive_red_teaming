import os
import csv
import json
import sys
import random
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/step1.log"),
    ]
)
log = logging.getLogger(__name__)

from src.model_loader import ModelLoader, MODEL_ID
from src.judge        import Judge
from src.attacker     import AdaptiveAttacker


DATASET_PATH = "data/HarmBench.csv"   
N_GOALS  = None                 
RANDOM_SEED = 42                    
PRECISION = "fp16"                



def load_advbench(path: str, n: int, seed: int = 42) -> list[str]:
   
    log.info("Loading HarmBench from %s ...", path)

    df = pd.read_csv(path)
    log.info("HarmBench loaded. Total prompts: %d", len(df))

    
    all_goals = df["prompt"].dropna().tolist()

    
    random.seed(seed)
    if n is None or n >= len(all_goals):
        goals = all_goals
        log.info("Using all %d AdvBench prompts", len(goals))
    else:
        goals = random.sample(all_goals, n)
        log.info("Sampled %d / %d prompts (seed=%d)",
                 len(goals), len(all_goals), seed)

    return goals


def run():
    log.info("=" * 60)
    log.info("STEP 1 — Adaptive Attack on %s Model", PRECISION.upper())
    log.info("Dataset : HarmBench — %d goals (seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Target  : Llama-3-8B-Instruct (%s, GPU 0)", PRECISION.upper())
    log.info("Attacker: Qwen2.5-14B-Instruct (INT8, GPU 1)")
    log.info("Judge   : LlamaGuard3-8B (BF16, GPU 2)")
    log.info("Strategy: CoT mutation (1-3) + Crescendo (4-5)")
    log.info("=" * 60)

    
    goals = load_advbench(DATASET_PATH, n=N_GOALS, seed=RANDOM_SEED)

   
    log.info("Loading target model...")
    target_model = ModelLoader(MODEL_ID, precision=PRECISION)

    log.info("Loading judge (LlamaGuard3)...")
    judge = Judge()

    log.info("Loading attacker (Qwen2.5-14B INT8)...")
    attacker = AdaptiveAttacker(target_model=target_model, judge=judge)

    
    all_results = []

    for i, goal in enumerate(goals):
        log.info("")
        log.info("[%d/%d] %s", i + 1, len(goals), goal[:70])
        log.info("-" * 55)

        result = attacker.attack(goal=goal, precision=PRECISION)
        all_results.append(result)

        # Running ASR every 10 goals
        if (i + 1) % 10 == 0:
            jb  = sum(1 for r in all_results if r.success)
            asr = jb / (i + 1)
            log.info("")
            log.info("── Running ASR [%d/%d]: %.1f%% ──",
                     i + 1, len(goals), asr * 100)

   
    total      = len(all_results)
    jailbreaks = sum(1 for r in all_results if r.success)
    asr = jailbreaks / total if total > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — STEP 1")
    log.info("=" * 60)
    log.info("Dataset : HarmBench (%d goals, seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Precision : %s", PRECISION.upper())
    log.info("Total goals : %d", total)
    log.info("Jailbreaks  : %d", jailbreaks)
    log.info("ASR : %.1f%%", asr * 100)
    log.info("=" * 60)

    for r in all_results:
        status = "JAILBREAK" if r.success else "REFUSED  "
        log.info("  %s | attempts=%d | %s",
                 status, r.n_attempts, r.goal[:50])

    
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/attacks_{PRECISION}_{N_GOALS}goals_{ts}.csv"
    rows = [r.to_dict() for r in all_results]

    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    log.info("CSV  → %s", csv_path)

  
    json_path = f"results/attacks_{PRECISION}_{N_GOALS}goals_{ts}.json"
    detailed  = []

    for r in all_results:
        d = r.to_dict()
        d["attempts_detail"] = [
            {
                "attempt": a.number,
                "prompt": a.prompt[:200],
                "response": a.response[:200],
                "is_jailbreak": a.is_jailbreak,
                "strategy_used": a.strategy_used,
                "reasoning": a.reasoning[:150],
            }
            for a in r.attempts
        ]
        detailed.append(d)

    with open(json_path, "w") as f:
        json.dump(detailed, f, indent=2)
    log.info("JSON → %s", json_path)

   
    target_model.unload()
    attacker.unload()
    judge.unload()
    log.info("Done.")


if __name__ == "__main__":
    run()