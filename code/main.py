import hydra
import easydict
import torch
import glob
import os


def parse(cfg):
    # device, stim_names, output_root, wav_dir, out_sr, pc, compile_torch, nfilts,
    args = cfg.env
    device = args.device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    stim_names = args.stim_names
    stim_names = stim_names.split(",")
    root = args.root  # /scratch/snormanh_lab/shared/projects
    project = args.project
    output_root = f"{root}/{project}/analysis"
    wav_dir = f"{root}/{project}/stimuli/stimulus_audio"
    if args.stim_names == "":
        # find stims using glob
        wav_paths = glob.glob(f"{wav_dir}/*.wav")
        stim_names = [
            os.path.splitext(os.path.basename(wav_path))[0] for wav_path in wav_paths
        ]
        stim_names.sort()

    args_dict = easydict.EasyDict(args)
    args_dict.update(
        {
            "device": device,
            "output_root": output_root,
            "stim_names": stim_names,
            "wav_dir": wav_dir,
        }
    )
    for key, value in args_dict.items():
        print(f"{key}: {value}")
    return args_dict


current_folder_path = os.path.dirname(__file__)
current_folder_absolute_path = os.path.abspath(current_folder_path)


@hydra.main(
    config_path=f"{current_folder_absolute_path}/conf",
    config_name="conf.yaml",
    version_base="1.3",
)
def main(cfg):
    args_dict = parse(cfg)
    if args_dict.feature == "ast":
        from utils.ast import ast

        ast(
            args_dict.device,
            args_dict.output_root,
            args_dict.stim_names,
            args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
        )
    elif args_dict.feature == "cochdnn":
        from utils.cochdnn import cochdnn

        cochdnn(
            args_dict.device,
            args_dict.output_root,
            args_dict.stim_names,
            args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
        )
    elif args_dict.feature == "cochresnet":
        from utils.cochresnet import cochresnet

        cochresnet(
            args_dict.device,
            args_dict.output_root,
            args_dict.stim_names,
            args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
        )
    elif args_dict.feature == "spectrogram":
        from utils.spectrogram import spectrogram

        spectrogram(
            args_dict.device,
            args_dict.output_root,
            args_dict.stim_names,
            args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            nfilts=args_dict.spectrogram.nfilts,
        )
        pass
    pass


if __name__ == "__main__":
    main()
