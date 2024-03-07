#!/bin/bash
#SBATCH -p doppelbock -t 04:00:00
#SBATCH -c 8
#SBATCH -a 0-4
#SBATCH --mem=128G
##SBATCH --gres=gpu:1 -x bhg0044,bhg0046,bhg0047,bhg0048
#SBATCH -o /scratch/snormanh_lab/shared/projects/intracranial-natsound165/analysis/logs/cs%a.txt
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=guoyang_liao@urmc.rochester.edu

# model_list=("ast" "cochdnn" "cochleagram" "cochresnet" "hubert" "spectrogram" "spectrotemporal")
# model=${model_list[$SLURM_ARRAY_TASK_ID]}
nonlins=("modulus" "modulus" "modulus" "real" "rect")
modulation_types=('tempmod' 'specmod' 'spectempmod' 'spectempmod' 'spectempmod')
model="cochleagram_spectrotemporal"
nonlin=${nonlins[$SLURM_ARRAY_TASK_ID]}
modulation_type=${modulation_types[$SLURM_ARRAY_TASK_ID]}

#如果$SLURM_ARRAY_TASK_ID不为0则睡眠10s
if [ $SLURM_ARRAY_TASK_ID -ne 0 ]; then
    sleep 10
fi

python /scratch/snormanh_lab/shared/code/snormanh_lab_python/feature_extraction/code/main.py "env=intracranial-natsound165" "env.feature=$model" "env.spectrotemporal.modulation_type=$modulation_type" "env.spectrotemporal.nonlin=$nonlin"