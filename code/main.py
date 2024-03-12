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
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)

    stim_names = args.stim_names
    stim_names = stim_names.split(",")
    root = args.root  # /scratch/snormanh_lab/shared/projects
    pca_weights_from = args.pca_weights_from
    pca_weights_from = (
        f"{root}/{pca_weights_from}/analysis" if pca_weights_from is not None else None
    )
    project = args.project
    output_root = f"{root}/{project}/analysis"
    wav_dir = f"{root}/{project}/stimuli/audio"
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
            "pca_weights_from": pca_weights_from,
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
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
    elif args_dict.feature == "cochdnn":
        from utils.cochdnn import cochdnn

        cochdnn(
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
    elif args_dict.feature == "cochleagram_spectrotemporal":
        from utils.cochleagram_spectrotemporal import cochleagram_spectrotemporal

        cochleagram_spectrotemporal(
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            modulation_type=args_dict.spectrotemporal.modulation_type,
            nonlin=args_dict.spectrotemporal.nonlin,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
    elif args_dict.feature == "cochresnet":
        from utils.cochresnet import cochresnet

        cochresnet(
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
    elif args_dict.feature == "hubert":
        from utils.hubert import hubert

        hubert(
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
    elif args_dict.feature == "spectrogram":
        from utils.spectrogram import spectrogram

        spectrogram(
            device=args_dict.device,
            output_root=args_dict.output_root,
            stim_names=args_dict.stim_names,
            wav_dir=args_dict.wav_dir,
            out_sr=args_dict.out_sr,
            pc=args_dict.pc,
            nfilts=args_dict.spectrogram.nfilts,
            time_window=args_dict.time_window,
            pca_weights_from=args_dict.pca_weights_from,
        )
        pass
    else:
        assert False, "Feature not implemented. Exiting."


if __name__ == "__main__":
    main()
