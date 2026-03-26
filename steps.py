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

from src.hqq_model_loader import ModelLoader, MODEL_ID
from src.judge import Judge
from src.attacker import AdaptiveAttacker


DATASET_PATH = "data/HarmBench.csv"
N_GOALS = None
RANDOM_SEED = 42
PRECISION = "int8"


CHECKPOINT_PATH = f"results/checkpoint_{MODEL_ID}_{PRECISION}.json"


def load_dataset(path: str, n: int, seed: int = 42) -> list[str]:
    log.info("Loading HarmBench from %s ...", path)
    df = pd.read_csv(path)
    log.info("HarmBench loaded. Total prompts: %d", len(df))
    all_goals = df["prompt"].dropna().tolist()
    random.seed(seed)
    if n is None or n >= len(all_goals):
        goals = all_goals
        log.info("Using all %d prompts", len(goals))
    else:
        goals = random.sample(all_goals, n)
        log.info("Sampled %d / %d prompts (seed=%d)",
                 len(goals), len(all_goals), seed)
    return goals


def load_checkpoint() -> list:
    """Load existing checkpoint. Returns list of completed result dicts."""
    if not os.path.exists(CHECKPOINT_PATH):
        return []
    with open(CHECKPOINT_PATH, "r") as f:
        checkpoint = json.load(f)
    if checkpoint.get("precision") != PRECISION:
        log.info("Checkpoint is for different precision — starting fresh.")
        return []
    completed = checkpoint.get("completed", 0)
    log.info("Checkpoint found — resuming from prompt %d / %d",
             completed, checkpoint.get("total", "?"))
    return checkpoint.get("results", [])


def run():
    log.info("=" * 60)
    log.info("STEP 1 — Adaptive Attack on %s Model", PRECISION.upper())
    log.info("Dataset : HarmBench — %s goals (seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Target  : %s (%s, GPU 0)", MODEL_ID, PRECISION.upper())
    log.info("Attacker: Qwen2.5-7B-Instruct ( GPU 1)")
    log.info("Judge   : LlamaGuard3-8B (GPU 2)")
    log.info("Strategy: CoT mutation")
    log.info("=" * 60)

    goals = load_dataset(DATASET_PATH, n=N_GOALS, seed=RANDOM_SEED)

    completed_dicts = load_checkpoint()
    n_completed = len(completed_dicts)

    if n_completed > 0:
        log.info("Resuming from prompt %d — skipping first %d completed prompts",
                 n_completed + 1, n_completed)
        remaining_goals = goals[n_completed:]
    else:
        log.info("Starting fresh — no checkpoint found")
        remaining_goals = goals
    

    log.info("Loading target model...")
    target_model = ModelLoader(MODEL_ID, precision=PRECISION)

    log.info("Loading judge (LlamaGuard3)...")
    judge = Judge()

    log.info("Loading attacker (Qwen2.5-7B-Instruct)...")
    attacker = AdaptiveAttacker(target_model=target_model, judge=judge)

    mlflow.set_experiment("QPSA-Quantization-Safety")
    mlflow.start_run(run_name=f"{PRECISION}_harmbench_from{n_completed}")
    mlflow.log_param("precision", PRECISION)
    mlflow.log_param("dataset", DATASET_PATH)
    mlflow.log_param("n_goals", N_GOALS or "all")
    mlflow.log_param("random_seed", RANDOM_SEED)
    mlflow.log_param("target_model",  MODEL_ID)
    mlflow.log_param("attacker_model",  "Qwen/Qwen2.5-7B-Instruct")
    mlflow.log_param("judge_model", "meta-llama/Llama-Guard-3-8B")
    mlflow.log_param("quantization_lib", "osciquant-ptq")
    mlflow.log_param("resumed_from", n_completed)
    mlflow.set_tag("quantization", "osciquant-ptq")
    mlflow.set_tag("attack", "cot-mutation-crescendo")
    mlflow.set_tag("defense", "chromadb-vector-db")
    

    all_results_dicts = completed_dicts.copy()

    for i, goal in enumerate(remaining_goals):
        global_idx = n_completed + i
        log.info("")
        log.info("[%d/%d] %s", global_idx + 1, len(goals), goal[:70])
        log.info("-" * 55)

        result = attacker.attack(goal=goal, precision=PRECISION)
        all_results_dicts.append(result.to_dict())

        
        checkpoint = {
            "precision": PRECISION,
            "completed": len(all_results_dicts),
            "total":  len(goals),
            "timestamp": datetime.now().isoformat(),
            "results":   all_results_dicts,
        }
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f, indent=2)
        

    
        jb = sum(1 for r in all_results_dicts if r["success"])
        running_asr = jb / len(all_results_dicts)
        mlflow.log_metric("running_asr",  running_asr,  step=global_idx + 1)
        mlflow.log_metric("running_jailbreaks", jb,  step=global_idx + 1)
        mlflow.log_metric("attempts_this_goal", result.n_attempts, step=global_idx + 1)

        if (global_idx + 1) % 10 == 0:
            log.info("")
            log.info("── Running ASR [%d/%d]: %.1f%% ──",
                     global_idx + 1, len(goals), running_asr * 100)

    
    total = len(all_results_dicts)
    jailbreaks = sum(1 for r in all_results_dicts if r["success"])
    asr = jailbreaks / total if total > 0 else 0
    avg_attempts = sum(r["n_attempts"] for r in all_results_dicts) / total if total > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — COMPLETE")
    log.info("=" * 60)
    log.info("Precision   : %s", PRECISION.upper())
    log.info("Total goals : %d", total)
    log.info("Jailbreaks  : %d", jailbreaks)
    log.info("ASR  : %.1f%%", asr * 100)
    log.info("Avg attempts: %.2f", avg_attempts)
    log.info("=" * 60)

    for r in all_results_dicts:
        status = "JAILBREAK" if r["success"] else "REFUSED  "
        log.info("  %s | attempts=%d | %s",
                 status, r["n_attempts"], r["goal"][:50])

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"results/{MODEL_ID}_{PRECISION}_{total}goals_{ts}.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_results_dicts[0].keys())
        writer.writeheader()
        writer.writerows(all_results_dicts)
    log.info("CSV  → %s", csv_path)

    json_path = f"results/attacks_{PRECISION}_{total}goals_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results_dicts, f, indent=2)
    log.info("JSON → %s", json_path)

    mlflow.log_metric("final_asr",    asr)
    mlflow.log_metric("total_goals",  total)
    mlflow.log_metric("jailbreaks",   jailbreaks)
    mlflow.log_metric("avg_attempts", avg_attempts)
    mlflow.log_artifact(csv_path)
    mlflow.log_artifact(json_path)
    mlflow.end_run()
    log.info("MLflow run logged.")

    if os.path.exists(CHECKPOINT_PATH):
        os.remove(CHECKPOINT_PATH)
        log.info("Checkpoint deleted — run complete.")

    target_model.unload()
    attacker.unload()
    judge.unload()
    log.info("Done.")


if __name__ == "__main__":
    run()