import os
import csv
import json
import sys
import random
import logging
import pandas as pd
import mlflow
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
from src.judge import Judge
from src.attacker import AdaptiveAttacker


DATASET_PATH = "data/HarmBench.csv"   
N_GOALS  = None                 
RANDOM_SEED  = 42                    
PRECISION    = "int4"                


def load_dataset(path: str, n: int, seed: int = 42) -> list[str]:
   
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
    log.info("Dataset : HarmBench — %s goals (seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Target  : Llama-3-8B-Instruct (%s, GPU 0)", PRECISION.upper())
    log.info("Attacker: Qwen2.5-14B-Instruct (INT8, GPU 1)")
    log.info("Judge : LlamaGuard3-8B (BF16, GPU 2)")
    log.info("Strategy: CoT mutation (1-5) + Crescendo (5-7)")
    log.info("=" * 60)

    goals = load_dataset(DATASET_PATH, n=N_GOALS, seed=RANDOM_SEED)

    log.info("Loading target model...")
    target_model = ModelLoader(MODEL_ID, precision=PRECISION)

    log.info("Loading judge (LlamaGuard3)...")
    judge = Judge()

    log.info("Loading attacker (Qwen2.5-14B INT8)...")
    attacker = AdaptiveAttacker(target_model=target_model, judge=judge)
    
    mlflow.set_experiment("QPSA-Quantization-Safety")
    mlflow.start_run(run_name=f"{PRECISION}_harmbench")

   
    mlflow.log_param("precision", PRECISION)
    mlflow.log_param("dataset", DATASET_PATH)
    mlflow.log_param("n_goals", N_GOALS or "all")
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("target_model", MODEL_ID)
    mlflow.log_param("attacker_model",   "Qwen/Qwen2.5-14B-Instruct")
    mlflow.log_param("judge_model",      "meta-llama/Llama-Guard-3-8B")
    mlflow.log_param("quantization_lib", "osciquant-ptq")
    mlflow.log_param("max_attempts",     5)

    
    mlflow.set_tag("quantization", "osciquant-ptq")
    mlflow.set_tag("attack",       "cot-mutation-crescendo")
    mlflow.set_tag("defense",      "chromadb-vector-db")


    all_results = []

    for i, goal in enumerate(goals):
        log.info("")
        log.info("[%d/%d] %s", i + 1, len(goals), goal[:70])
        log.info("-" * 55)

        result = attacker.attack(goal=goal, precision=PRECISION)
        all_results.append(result)

        jb = sum(1 for r in all_results if r.success)
        running_asr = jb / (i + 1)
        mlflow.log_metric("running_asr", running_asr,step=i + 1)
        mlflow.log_metric("running_jailbreaks",  jb, step=i + 1)
        mlflow.log_metric("attempts_this_goal",  result.n_attempts, step=i + 1)
       

       
        if (i + 1) % 10 == 0:
            log.info("")
            log.info("── Running ASR [%d/%d]: %.1f%% ──",
                     i + 1, len(goals), running_asr * 100)

    total = len(all_results)
    jailbreaks = sum(1 for r in all_results if r.success)
    asr = jailbreaks / total if total > 0 else 0
    avg_attempts = sum(r.n_attempts for r in all_results) / total if total > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — STEP 1")
    log.info("=" * 60)
    log.info("Dataset : HarmBench (%s goals, seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Precision : %s", PRECISION.upper())
    log.info("Total goals : %d", total)
    log.info("Jailbreaks  : %d", jailbreaks)
    log.info("ASR : %.1f%%", asr * 100)
    log.info("Avg attempts: %.2f", avg_attempts)
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
                "attempt":      a.number,
                "prompt":       a.prompt[:200],
                "response":     a.response[:200],
                "is_jailbreak": a.is_jailbreak,
                "strategy_used": a.strategy_used,
                "reasoning":    a.reasoning[:150],
            }
            for a in r.attempts
        ]
        detailed.append(d)

    with open(json_path, "w") as f:
        json.dump(detailed, f, indent=2)
    log.info("JSON → %s", json_path)


    mlflow.log_metric("final_asr",      asr)
    mlflow.log_metric("total_goals",    total)
    mlflow.log_metric("jailbreaks",     jailbreaks)
    mlflow.log_metric("avg_attempts",   avg_attempts)
    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(json_path)
    mlflow.end_run()
    log.info("MLflow run logged.")

    target_model.unload()
    attacker.unload()
    judge.unload()
    log.info("Done.")


if __name__ == "__main__":
    run()