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
        logging.FileHandler("logs/base_evaluation.log"),
    ]
)
log = logging.getLogger(__name__)

from src.hqq_model_loader import ModelLoader, MODEL_ID
from src.qwen_judge import Judge  

DATASET_PATH = "data/HarmBench.csv"
N_GOALS = None
RANDOM_SEED = 42
PRECISION = "int4"

CHECKPOINT_PATH = f"results/checkpoint_base_{PRECISION}.json"

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

def get_short_model_name(model_id: str) -> str:
    name = model_id.split("/")[-1] if "/" in model_id else model_id
    name = name.lower()
    for suffix in ["-instruct", "-it", "-chat"]:
        name = name.replace(suffix, "")
    return name

def run():
    log.info("=" * 60)
    log.info("BASE EVALUATION — %s Model", PRECISION.upper())
    log.info("Dataset : HarmBench — %s goals (seed=%d)", N_GOALS, RANDOM_SEED)
    log.info("Target  : %s (%s)", MODEL_ID, PRECISION.upper())
    log.info("Judge   : Qwen3Guard-Gen-8B (response only)")
    log.info("Strategy: Direct base prompt evaluation (no attacks)")
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

    target_device = "cuda:0"
    judge_device = "cuda:0"
    
    log.info(f"Device assignment → Target: {target_device} | Judge: {judge_device}")

    log.info("Loading target model...")
    target_model = ModelLoader(MODEL_ID, precision=PRECISION, device=target_device)

    log.info("Loading judge (Qwen3Guard-Gen-8B)...")
    judge = Judge(device=judge_device)

    all_results_dicts = completed_dicts.copy()

    for i, goal in enumerate(remaining_goals):
        global_idx = n_completed + i
        log.info("")
        log.info("[%d/%d] %s", global_idx + 1, len(goals), goal[:70])
        log.info("-" * 55)

        response = target_model.generate(goal)
        log.info("  Response: %s...", response[:100])

        verdict = judge.evaluate(response=response, goal=goal)

        status = "Jailbroken" if verdict.is_jailbreak else "Not jailbroken"
        log.info("  %s", status)
        log.info("  Judge: safety=%s refusal=%s", verdict.raw_label, verdict.refusal_label)
        if verdict.harm_name and verdict.harm_name != "None":
            log.info("  Harm category: %s (%s)", verdict.harm_category, verdict.harm_name)
        log.info("  %s", status)
        log.info("  All attempts exhausted. %s.", status.lower())

        result_dict = {
            "goal": goal,
            "precision": PRECISION,
            "success": verdict.is_jailbreak,
            "n_attempts": 1,
            "harm_category": verdict.harm_category,
            "harm_name": verdict.harm_name,
            "raw_label": verdict.raw_label,
            "refusal_label": verdict.refusal_label,
            "final_prompt": goal,
            "final_response": response[:300],
            "judge_safety_label": verdict.raw_label,
            "judge_refusal": verdict.refusal_label,
            "judge_harm_category": verdict.harm_category,
            "judge_harm_name": verdict.harm_name,
        }
        all_results_dicts.append(result_dict)

        checkpoint = {
            "precision": PRECISION,
            "completed": len(all_results_dicts),
            "total": len(goals),
            "timestamp": datetime.now().isoformat(),
            "results": all_results_dicts,
        }
        with open(CHECKPOINT_PATH, "w") as f:
            json.dump(checkpoint, f, indent=2)

        if (global_idx + 1) % 10 == 0:
            total_checked = len(all_results_dicts)
            jb = sum(1 for r in all_results_dicts if r["success"])
            cont = sum(1 for r in all_results_dicts if r.get("judge_safety_label") == "Controversial")
            ref = sum(1 for r in all_results_dicts if r.get("judge_refusal") == "Yes")
            
            log.info("")
            log.info("── Running Stats [%d/%d] ──", global_idx + 1, len(goals))
            log.info("  ASR: %.1f%% | Controversial: %.1f%% | Refusal: %.1f%%",
                     (jb / total_checked) * 100,
                     (cont / total_checked) * 100,
                     (ref / total_checked) * 100)
            log.info("────────────────────────────")

    total = len(all_results_dicts)
    jailbreaks = sum(1 for r in all_results_dicts if r["success"])
    asr = jailbreaks / total if total > 0 else 0
    avg_attempts = sum(r["n_attempts"] for r in all_results_dicts) / total if total > 0 else 0

    controversial_count = sum(1 for r in all_results_dicts if r.get("judge_safety_label") == "Controversial")
    controversial_rate = controversial_count / total if total > 0 else 0
    safe_count = sum(1 for r in all_results_dicts if r.get("judge_safety_label") == "Safe")
    unsafe_count = sum(1 for r in all_results_dicts if r.get("judge_safety_label") == "Unsafe")
    unknown_count = sum(
        1 for r in all_results_dicts
        if r.get("judge_safety_label") not in {"Safe", "Unsafe", "Controversial"}
    )
    unknown_rate = unknown_count / total if total > 0 else 0
    refused_count = sum(1 for r in all_results_dicts if r.get("judge_refusal") == "Yes")
    refusal_rate = refused_count / total if total > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("RESULTS — COMPLETE")
    log.info("=" * 60)
    log.info("Precision       : %s", PRECISION.upper())
    log.info("Total goals     : %d", total)
    log.info("-" * 40)
    log.info("Jailbreaks    : %d (%.1f%% ASR)", jailbreaks, asr * 100)
    log.info("Safety Labels :")
    log.info("• Safe         : %d (%.1f%%)", safe_count, safe_count/total*100 if total else 0)
    log.info(" • Unsafe       : %d (%.1f%%)", unsafe_count, unsafe_count/total*100 if total else 0)
    log.info(" • Controversial: %d (%.1f%%)", controversial_count, controversial_rate * 100)
    log.info(" • Unknown      : %d (%.1f%%)", unknown_count, unknown_rate * 100)
    log.info("Refusal Rate  : %d (%.1f%%)", refused_count, refusal_rate * 100)
    log.info("📊Avg Attempts  : %.2f", avg_attempts)
    log.info("=" * 60)

    for r in all_results_dicts:
        status = "JAILBREAK" if r["success"] else "REFUSED  "
        harm = r.get("judge_harm_name", r.get("harm_category", "N/A"))
        log.info("  %s | attempts=%d | harm=%-25s | %s",
                 status, r["n_attempts"], harm, r["goal"][:40])

    short_name = get_short_model_name(MODEL_ID)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_name = Path(DATASET_PATH).stem
    sample_info = f"{N_GOALS}sample" if N_GOALS else f"{total}goals"
    
    csv_path = f"results/{short_name}_{PRECISION}_{dataset_name}_{sample_info}_{ts}.csv"
    fieldnames = list(all_results_dicts[0].keys())
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_results_dicts)
    log.info("CSV  → %s", csv_path)

    json_path = f"results/attacks_{short_name}_{PRECISION}_{total}goals_{ts}.json"
    with open(json_path, "w") as f:
        json.dump(all_results_dicts, f, indent=2)
    log.info("JSON → %s", json_path)

    if os.path.exists(CHECKPOINT_PATH):
        try:
            os.remove(CHECKPOINT_PATH)
            log.info("Checkpoint deleted — run complete.")
        except OSError as e:
            log.warning("Could not delete checkpoint: %s", e)

    target_model.unload()
    judge.unload()
    log.info("Done.")


if __name__ == "__main__":
    run()