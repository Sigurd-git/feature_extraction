#!/bin/bash
#SBATCH -p preempt -t 04:00:00
#SBATCH -c 20
#SBATCH -a 0-4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/syllable-invariances_v2/analysis/logs/cs%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

# model_list=("ast" "cochdnn" "cochleagram" "cochresnet" "hubert" "spectrogram" "spectrotemporal")
# model=${model_list[$SLURM_ARRAY_TASK_ID]}
nonlins=("modulus" "modulus" "modulus" "real" "rect")
modulation_types=('tempmod' 'specmod' 'spectempmod' 'spectempmod' 'spectempmod')
model="cochleagram_spectrotemporal"
nonlin=${nonlins[$SLURM_ARRAY_TASK_ID]}
modulation_type=${modulation_types[$SLURM_ARRAY_TASK_ID]}
python /home/gliao2/samlab_Sigurd/feature_extration/code/main.py "env=syllable-invariances_v2" "env.feature=$model" "env.spectrotemporal.modulation_type=$modulation_type" "env.spectrotemporal.nonlin=$nonlin"