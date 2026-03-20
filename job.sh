#!/bin/bash
#SBATCH --job-name=adaptive_attack
#SBATCH --partition=l4
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:3
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/step1_%j.out
#SBATCH --error=logs/step1_%j.err
#SBATCH --mail-type=FAIL

echo "============================================"
echo "Job ID   : $SLURM_JOB_ID"
echo "Node     : $SLURMD_NODENAME"
echo "Started  : $(date)"
echo "============================================"
 
nvidia-smi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_safety
 
echo "Python   : $(which python3)"
echo "Python V : $(python3 --version)"


echo "Checking imports..."
python3 -c "
import sys, os
sys.path.insert(0, os.path.expanduser('~/adaptive_red_teaming'))
from src.model_loader import ModelLoader
from src.judge        import Judge
from src.attacker     import AdaptiveAttacker
print('All imports OK')
"

if [ $? -ne 0 ]; then
    echo "ERROR: Import check failed — aborting"
    exit 1
fi

cd ~/adaptive_red_teaming
python3 steps.py
 
echo "============================================"
echo "Finished : $(date)"
echo "============================================"
