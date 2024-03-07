#!/bin/bash
#SBATCH -p doppelbock -t 04:00:00
#SBATCH -c 20
#SBATCH -a 0
#SBATCH --mem=128G
#SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/syllable-invariances_v1/analysis/logs/cs%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu




# model_list=("ast" "cochdnn" "cochresnet" "hubert" "spectrogram")
# model=${model_list[$SLURM_ARRAY_TASK_ID]}
# python /scratch/snormanh_lab/shared/code/snormanh_lab_python/feature_extraction/code/main.py "env=syllable-invariances_v1" "env.feature=$model"

non_list=(modulus real rect)
modulation_list=(spectempmod specmod tempmod)
length_modulation_list=${#modulation_list[@]}
modulation_index=$(( SLURM_ARRAY_TASK_ID % length_modulation_list))
non_list_index=$(( SLURM_ARRAY_TASK_ID / length_modulation_list))
modulation_type=${modulation_list[modulation_index]}
non_lin=${non_list[non_list_index]}
model=cochleagram_spectrotemporal

python /scratch/snormanh_lab/shared/code/snormanh_lab_python/feature_extraction/code/main.py "env=syllable-invariances_v1" "env.feature=$model" "env.spectrotemporal.modulation_type=$modulation_type" "env.spectrotemporal.nonlin=$non_lin"