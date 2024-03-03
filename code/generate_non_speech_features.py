
import torch
import torchaudio
from general_analysis_code.get_model_output import deepspeech_encoder
import glob
# import matlab.engine

from utils.feature_extraction import generate_non_speech_feature_set_for_one_stim,generate_deepspeech2_features

######personalize parameters
device = torch.device("cuda:0")
compile_torch=True

encoder_deepspeech = deepspeech_encoder("/home/gliao2/samlab_Sigurd/feature_extration/code/utils/deepspeech2-pretrained.ckpt", device=device,compile_torch=compile_torch)

# HUBERT_bundle = torchaudio.pipelines.HUBERT_BASE

# HUBERT_model = HUBERT_bundle.get_model().to(device)

# wav2vec2_bundle = torchaudio.pipelines.WAV2VEC2_BASE  # or WAV2VEC2_BASE

# wav2vec2_model = wav2vec2_bundle.get_model().to(device)

# if compile_torch:
#     HUBERT_model = torch.compile(HUBERT_model)
#     wav2vec2_model = torch.compile(wav2vec2_model)
# eng = matlab.engine.start_matlab()
# eng.addpath(eng.genpath('/home/gliao2/samlab_Sigurd/feature_extration/code/utils'))






projects_set = ('columbia-localizer','ecog-TCI-completion','envsound-invariance','natural-story-variation','naturalsound-iEEG-sanitized-mixtures','nima-lab-localizer-v2','omts-iEEG','speech-completion-encoding-iEEG','speech-TCI')

for project in projects_set:
    output_root = f"/scratch/snormanh_lab/shared/projects/{project}/analysis"
    wav_dir = f"/scratch/snormanh_lab/shared/projects/{project}/stimuli/stimulus_audio"
    wav_paths = glob.glob(f"{wav_dir}/*.wav")
    for wav_path in wav_paths:
        # generate_non_speech_feature_set_for_one_stim(device, HUBERT_model, wav2vec2_model,encoder_deepspeech, eng, wav_path, output_root,n_t=None)
        # generate deepspeech2 features
        generate_deepspeech2_features(wav_path, output_root,encoder_deepspeech,n_t=None)
    

