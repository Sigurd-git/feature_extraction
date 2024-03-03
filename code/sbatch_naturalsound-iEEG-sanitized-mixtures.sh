#!/bin/bash
#SBATCH -p doppelbock -t 04:00:00
#SBATCH -c 20
#SBATCH -a 0-3
#SBATCH --mem=250G
#SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/naturalsound-iEEG-sanitized-mixtures/analysis/logs/%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

model_list=("ast" "cochdnn" "cochresnet" "hubert")
model=${model_list[$SLURM_ARRAY_TASK_ID]}
python /home/gliao2/samlab_Sigurd/feature_extration/code/main.py "env=naturalsound-iEEG-sanitized-mixtures" "env.feature=$model"