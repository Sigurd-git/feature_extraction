#!/bin/bash
#SBATCH -p preempt -t 02:00:00
#SBATCH -c 8
#SBATCH -a 0-13
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/syllable-invariances_v2/analysis/logs/%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

model_list=("ast" "cochdnn" "cochleagram" "cochresnet" "hubert" "spectrogram" "spectrotemporal")