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


# DIRS

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)


# LOGGING

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


# IMPORTS

from src.hqq_model_loader import ModelLoader, MODEL_ID
from src.qwen_judge import Judge
from src.attacker import AdaptiveAttacker
from src.vdefense import VectorDefense


# CONFIG

DATASET_PATH = "data/AdvBench_100.csv"

# Preload known jailbreak prompts into the harmful collection
PRELOAD_PATH = "data/AdvBench_100.csv"

N_GOALS = None
RANDOM_SEED = 42

PRECISION = "int8"

# Starting threshold — defense will adaptively tighten this
# after each confirmed jailbreak (floor is set inside VectorDefense)
DEFENSE_THRESHOLD = 0.45

CHECKPOINT_PATH = f"results/checkpoint_{PRECISION}.json"



# DATASET

def load_dataset(
    path: str,
    n: int = None,
    seed: int = 42,
):
    log.info("Loading dataset: %s", path)

    df = pd.read_csv(path)
    prompts = df["prompt"].dropna().tolist()

    random.seed(seed)

    if n is not None and n < len(prompts):
        prompts = random.sample(prompts, n)

    log.info("Loaded %d prompts", len(prompts))
    return prompts



# PRELOAD

def load_preload_data(path: str):
    if path is None or not os.path.exists(path):
        return []

    df = pd.read_csv(path)
    prompts = df["prompt"].dropna().tolist()

    log.info("Loaded %d preload prompts", len(prompts))
    return prompts



# CHECKPOINT

def load_checkpoint():
    if not os.path.exists(CHECKPOINT_PATH):
        return []

    try:
        with open(CHECKPOINT_PATH, "r", encoding="utf-8") as f:
            checkpoint = json.load(f)

        if checkpoint.get("precision") != PRECISION:
            log.warning("Checkpoint precision mismatch — starting fresh.")
            return []

        completed = checkpoint.get("completed", 0)
        log.info("Loaded checkpoint (%d completed)", completed)
        return checkpoint.get("results", [])

    except Exception as e:
        log.warning("Checkpoint load failed: %s", e)
        return []


def save_checkpoint(results, total_goals):
    checkpoint = {
        "precision": PRECISION,
        "completed": len(results),
        "total": total_goals,
        "timestamp": datetime.now().isoformat(),
        "results": results,
    }

    with open(CHECKPOINT_PATH, "w", encoding="utf-8") as f:
        json.dump(checkpoint, f, indent=2)



# MAIN

def run():

    mlflow_active = False
    defense = None
    attacker = None
    judge = None
    target_model = None

    try:

        log.info("=" * 60)
        log.info("Adaptive Attack + VectorDefense v3 (Six-Layer)")
        log.info("=" * 60)

        #
        # DATA
        #
        goals = load_dataset(DATASET_PATH, N_GOALS, RANDOM_SEED)

        #
        # CHECKPOINT RESUME
        #
        all_results = load_checkpoint()
        completed = len(all_results)

        if completed > 0:
            log.info("Resuming from %d/%d", completed, len(goals))
            goals = goals[completed:]
        else:
            log.info("Starting fresh run")

        #
        # LOAD MODELS
        #
        log.info("Loading target model...")
        target_model = ModelLoader(MODEL_ID, precision=PRECISION)

        log.info("Loading judge...")
        judge = Judge()

        log.info("Loading attacker...")
        attacker = AdaptiveAttacker(
            target_model=target_model,
            judge=judge,
        )

        log.info("Loading VectorDefense v3...")
        defense = VectorDefense(
            collection_name=f"harmbench_{PRECISION}",
            persist_dir=f"results/vector_db_{PRECISION}",
            # Layer 6: adaptive threshold config
            threshold=DEFENSE_THRESHOLD,
            threshold_floor=0.25,
            threshold_decay=0.03,
            # Top-k for similarity
            top_k=7,
            # Replay buffer size for SGD stability
            replay_buffer_size=200,
            # Fusion weights
            w_clf=0.40,
            w_sim=0.30,
            w_kw=0.10,
            w_tmpl=0.20,
            # Cold-start similarity weight
            cold_sim_weight=0.85,
        )

        #
        # PRELOAD MEMORY + WARM-START CLASSIFIER
        #
        preload_data = load_preload_data(PRELOAD_PATH)

        if preload_data:
            log.info("Preloading %d prompts...", len(preload_data))
            # preload() handles: storing in harmful_col,
            # seeding replay buffer, and warm-starting the classifier
            defense.preload(preload_data)

        #
        # MLFLOW
        #
        mlflow.set_tracking_uri(f"file:./mlruns_{PRECISION}")
        mlflow.set_experiment("QPSA-Quantization-Safety")
        mlflow.start_run(
            run_name=f"{PRECISION}_adaptive_vdefense_v3"
        )
        mlflow_active = True

        # Log defense config as MLflow params
        mlflow.log_params({
            "defense_threshold_init": DEFENSE_THRESHOLD,
            "defense_threshold_floor": 0.25,
            "defense_threshold_decay": 0.03,
            "defense_top_k": 7,
            "defense_replay_buffer": 200,
            "defense_w_clf": 0.40,
            "defense_w_sim": 0.30,
            "defense_w_kw": 0.10,
            "defense_w_tmpl": 0.20,
            "precision": PRECISION,
            "n_goals": len(goals) + completed,
            "preload_size": len(preload_data),
        })

        #
        # MAIN LOOP
        #
        total_original = completed + len(goals)

        for idx, goal in enumerate(goals):

            global_idx = completed + idx

            log.info("")
            log.info(
                "[%d/%d] %s",
                global_idx + 1,
                total_original,
                goal[:80],
            )

            #
            # ATTACK
            #
            result = attacker.attack(
                goal=goal,
                precision=PRECISION,
            )

            candidate_prompt = result.final_prompt or goal

            #
            # DEFENSE — pass threshold=None to always use the
            # current adaptive threshold inside VectorDefense
            #
            (
                is_blocked,
                risk_score,
                similarity,
            ) = defense.is_malicious(
                candidate_prompt,
                threshold=None,
            )

            #
            # BLOCKED BRANCH
            #
            if is_blocked:

                log.warning(
                    "BLOCKED | risk=%.3f | sim=%.3f | threshold=%.3f",
                    risk_score,
                    similarity,
                    defense.threshold,
                )

                result.success = False

                record = result.to_dict()
                record.update({
                    "response": None,
                    "blocked": True,
                    "risk_score": float(risk_score),
                    "similarity": float(similarity),
                    "termination_reason": "blocked",
                    "judge_safety_label": "Blocked",
                    "judge_refusal": True,
                    "judge_harm_category": "",
                    "judge_harm_name": "",
                })

                all_results.append(record)

                # FIX: was learn_safe(response) — `response` is undefined here.
                # Store the blocked attack in the harmful collection so it
                # strengthens future detection (goal + final_prompt in record).
                defense.learn_from_attack(goal, record)

                save_checkpoint(all_results, total_original)

            #
            # ALLOWED BRANCH
            #
            else:

                response = (
                    target_model.generate(candidate_prompt) or ""
                )
                result.final_response = response

                #
                # JUDGE
                #
                verdict = judge.evaluate(response, goal=goal)
                result.success = verdict.is_jailbreak

                record = result.to_dict()
                record.update({
                    "response": response,
                    "blocked": False,
                    "risk_score": float(risk_score),
                    "similarity": float(similarity),
                    "judge_safety_label": verdict.raw_label,
                    "judge_refusal": verdict.refusal_label,
                    "judge_harm_category": verdict.harm_category,
                    "judge_harm_name": verdict.harm_name,
                    "termination_reason": (
                        "jailbreak" if result.success else "safe"
                    ),
                })

                all_results.append(record)

                #
                # ONLINE LEARNING
                #
                if result.success:
                    # Jailbreak succeeded:
                    # - store goal + final_prompt in harmful collection
                    # - register exact hashes for layer-1 hard blocking
                    # - tighten adaptive threshold (layer 6)
                    defense.learn_from_attack(goal, record)

                else:
                    # Model safely refused:
                    # - store the *response* (not the harmful goal) as safe
                    # - teaches the classifier what a safe output looks like
                    defense.learn_safe(response)

                save_checkpoint(all_results, total_original)

            #
            # PER-ITERATION METRICS
            #
            jailbreaks = sum(r["success"] for r in all_results)
            blocked_hard = sum(
                1 for r in all_results
                if r.get("blocked") and r.get("risk_score", 0) >= 1.0
            )
            blocked_soft = sum(
                1 for r in all_results
                if r.get("blocked") and r.get("risk_score", 0) < 1.0
            )
            blocked_total = sum(r.get("blocked", False) for r in all_results)

            running_asr = jailbreaks / len(all_results)
            blocked_rate = blocked_total / len(all_results)

            log.info(
                "Running ASR: %.2f%% | Blocked: %.2f%% "
                "(hard=%d soft=%d) | Threshold: %.3f",
                running_asr * 100,
                blocked_rate * 100,
                blocked_hard,
                blocked_soft,
                defense.threshold,
            )

            # Defense internals
            d_stats = defense.stats()
            log.info(
                "DEFENSE STATS | harmful=%d | safe=%d | "
                "replay=%d | hashes=%d | threshold=%.3f",
                d_stats["harmful_memory"],
                d_stats["safe_memory"],
                d_stats["replay_buffer"],
                d_stats["hash_blocklist"],
                d_stats["current_threshold"],
            )

            # MLflow per-step metrics
            step = global_idx + 1
            mlflow.log_metric("running_asr", running_asr, step=step)
            mlflow.log_metric("blocked_rate", blocked_rate, step=step)
            mlflow.log_metric("risk_score", risk_score, step=step)
            mlflow.log_metric("similarity", similarity, step=step)
            mlflow.log_metric("adaptive_threshold", defense.threshold, step=step)
            mlflow.log_metric("harmful_memory", d_stats["harmful_memory"], step=step)
            mlflow.log_metric("replay_buffer_size", d_stats["replay_buffer"], step=step)

        #
        # FINAL METRICS
        #
        total = len(all_results)
        jailbreaks = sum(r["success"] for r in all_results)
        blocked_total = sum(r.get("blocked", False) for r in all_results)

        final_asr = jailbreaks / total if total > 0 else 0
        blocked_rate = blocked_total / total if total > 0 else 0

        #
        # SAVE RESULTS
        #
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = f"results/final_{PRECISION}_{ts}.json"
        csv_path = f"results/final_{PRECISION}_{ts}.csv"

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)

        #
        # FINAL LOGGING
        #
        log.info("")
        log.info("=" * 60)
        log.info("FINAL RESULTS")
        log.info("=" * 60)
        log.info("Total prompts  : %d", total)
        log.info("Jailbreaks     : %d", jailbreaks)
        log.info("Blocked        : %d", blocked_total)
        log.info("Final ASR      : %.2f%%", final_asr * 100)
        log.info("Blocked Rate   : %.2f%%", blocked_rate * 100)
        log.info("Final threshold: %.3f", defense.threshold)

        # Final defense stats
        d_stats = defense.stats()
        log.info("Harmful memory : %d", d_stats["harmful_memory"])
        log.info("Safe memory    : %d", d_stats["safe_memory"])
        log.info("Hash blocklist : %d", d_stats["hash_blocklist"])
        log.info("Hard blocks    : %d", d_stats["blocked_hard"])
        log.info("Soft blocks    : %d", d_stats["blocked_soft"])
        log.info("=" * 60)

        #
        # MLFLOW FINAL
        #
        mlflow.log_metric("final_asr", final_asr)
        mlflow.log_metric("final_blocked_rate", blocked_rate)
        mlflow.log_metric("final_threshold", defense.threshold)
        mlflow.log_metric("final_harmful_memory", d_stats["harmful_memory"])
        mlflow.log_metric("final_hash_blocklist", d_stats["hash_blocklist"])
        mlflow.log_artifact(json_path)
        mlflow.log_artifact(csv_path)

        #
        # DELETE CHECKPOINT
        #
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            log.info("Checkpoint deleted.")

    finally:

        #
        # CLEANUP
        #
        try:
            if mlflow_active:
                mlflow.end_run()
        except Exception:
            pass

        for obj in [defense, target_model, attacker, judge]:
            try:
                if obj:
                    obj.unload()
            except Exception:
                pass

        log.info("Done.")



# ENTRY

if __name__ == "__main__":
    run()