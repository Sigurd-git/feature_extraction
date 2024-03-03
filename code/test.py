from transformers import AutoModel
from utils.ast import generate_AST_features
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(device)
output_root = "/scratch/snormanh_lab/shared/projects/intracranial-natsound119/analysis"

generate_AST_features(
    device,
    model,
    "/scratch/snormanh_lab/shared/projects/intracranial-natsound119/stimuli/stimulus_audio/aninonvoc_cat83_rec1_cicadas_excerpt1.wav",
    output_root,
    out_sr=100,
)

pass
