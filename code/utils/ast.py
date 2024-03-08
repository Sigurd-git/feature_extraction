import os
import hdf5storage
import numpy as np
import torch
import torchaudio
from general_analysis_code.preprocess import align_time
from transformers import AutoProcessor, AutoModel
from collections import defaultdict
import librosa
from utils.pc import generate_pc
from feature_extraction.code.utils.shared import write_summary


def ast(device, output_root, stim_names, wav_dir, out_sr=100, pc=100, **kwargs):
    AST_model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(
        device
    )

    for stim_index, stim_name in enumerate(stim_names):
        wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
        generate_AST_features(
            device, AST_model, wav_path, output_root, n_t=None, out_sr=out_sr
        )

    # compute PC of ast features
    feature_name = "ast"
    layer_num = 14
    for layer in range(layer_num):
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            feature_path = (
                f"{output_root}/features/{feature_name}/layer{layer}/{stim_name}.mat"
            )
            feature = hdf5storage.loadmat(feature_path)["features"]
            wav_features.append(feature)
        if pc is not None:
            pca_pipeline = generate_pc(
                wav_features,
                pc,
                output_root,
                feature_name,
                stim_names,
                demean=True,
                std=False,
                variant=f"layer{layer}",
            )
            generate_pc(
                wav_features,
                pc,
                output_root,
                feature_name,
                stim_names,
                demean=True,
                std=False,
                variant=f"layer{layer}",
                pca_pipeline=pca_pipeline,
            )


def generate_AST_features(
    device, AST_model, wav_path, output_root, n_t=None, out_sr=100, time_window=[-1, 1]
):
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "ast")

    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)

    waveform, sample_rate = torchaudio.load(wav_path)
    t_num_new = int(np.round(waveform.shape[1] / sample_rate * out_sr))
    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr
    # pad the waveform to cover 0
    t_0s = [0.01246875] + [0.08746875] * 13
    srs = [100] + [10] * 13
    before_pad_number = int(np.max(t_0s) * sample_rate) + 1
    after_pad_number = 160000
    t_0s = np.array(t_0s) - before_pad_number / sample_rate
    waveform = torch.nn.functional.pad(waveform, (before_pad_number, after_pad_number))
    outputs = extract_AST(device, AST_model, waveform, sample_rate)

    for i, (feats, t_0, sr) in enumerate(zip(outputs, t_0s, srs)):
        # The start time of output of layer n is: t_0_{n+1} = t_0_{n} + (conv_kernel_{n}-1) / (2*sr_{n})
        # The sr of output of layer n is: sr_{n+1} = sr_{n} / conv_stride_{n}
        print(f"t_0_{i}={t_0}, sr_{i}={sr}")
        feature_variant_out_dir = os.path.join(feature_class_out_dir, f"layer{i}")
        if not os.path.exists(feature_variant_out_dir):
            os.makedirs(feature_variant_out_dir)
        feats = feats.cpu().numpy()
        ### align time to start from 0 and make the length to be n_t
        t_length = feats.shape[0]
        t_origin = np.arange(t_length) / sr + t_0

        feats = align_time(feats, t_origin, t_new, "t f")
        print(f"Feature {i}: {feats.shape}")

        # save each layer as mat
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")

        # save data as mat
        hdf5storage.savemat(out_mat_path, {"features": feats, "t": t_new})
        # generate meta file for this layer

        write_summary(
            feature_variant_out_dir,
            time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
            dimensions="[time, feature]",
            extra="Nothing",
        )


def process_waveform(waveform, kernel=163840, stride=81920):
    if len(waveform) <= stride:
        # pad the waveform
        waveform = torch.nn.functional.pad(waveform, (0, kernel - len(waveform)))
        processed_waveforms = [waveform]
        return processed_waveforms
    else:
        if len(waveform) < kernel + stride:
            waveform = torch.nn.functional.pad(
                waveform, (0, kernel + stride - len(waveform))
            )
        waveform_np = (
            waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        )
        frames = librosa.util.frame(
            waveform_np, frame_length=kernel, hop_length=stride
        ).T
        processed_waveforms = [torch.from_numpy(frame) for frame in frames]

    return processed_waveforms


def extract_AST(device, AST_model, waveform, sample_rate, stride_in_seconds=9):
    """
    Extract features from an audio signal using an Audio Spectrogram Transformer (AST) model.

    This function preprocesses the waveform by resampling it to a target sample rate, processes it
    through a pretrained AST model to extract features, and then organizes these features based
    on specified stride lengths in seconds for further analysis or downstream tasks.

    Args:
        device: The device (CPU/GPU) to run the inference on.
        AST_model: The pretrained Audio Spectrogram Transformer model.
        waveform: The input audio signal as a tensor.
        sample_rate: The sample rate of the input audio signal.
        stride_in_seconds: The stride length in seconds for segmenting the audio signal.

    Returns:
        A list of tensors containing the processed features from the AST model, organized by the
        specified stride length and sample rates for different layers of the model.
    """

    # Define the target sample rate for the model input.
    model_input_sample_rate = 16000

    # Sample rates for different layers of the model.
    srs = [100] + [10] * 13

    # Convert waveform to tensor if it's not already.
    waveform = torch.as_tensor(waveform).reshape(-1)

    # Calculate the length of the waveform in seconds.
    wav_length_in_seconds = waveform.shape[0] / sample_rate

    # Resample the waveform to the target sample rate if necessary.
    if sample_rate != model_input_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, model_input_sample_rate
        )

    # Calculate the stride in samples, converting stride_in_seconds to samples.
    stride = model_input_sample_rate * stride_in_seconds

    # Initialize the processor with the pretrained model.
    processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

    # Process the waveform into segments based on the calculated stride.
    processed_waveforms = process_waveform(waveform, stride=stride)

    # Initialize a dictionary to hold the outputs for different segments.
    output_dict = defaultdict(list)

    # Process each segmented waveform through the AST model.
    for waveform in processed_waveforms:
        # Prepare the waveform for the model.
        inputs = processor(
            raw_speech=waveform,
            return_tensors="pt",
            sampling_rate=model_input_sample_rate,
        ).to(device)

        # Perform inference without gradient calculation.
        with torch.inference_mode():
            outputs = AST_model(**inputs, output_hidden_states=True)
            activations = outputs.hidden_states

        # Process and reshape the model outputs for each layer.
        outputs = [inputs["input_values"].squeeze(0)] + [
            activation[0, 2:, :].reshape(12, 101, -1).permute(1, 0, 2).reshape(101, -1)
            for activation in activations
        ]

        # Crop and append the outputs based on the stride for each layer.
        for index, output in enumerate(outputs):
            sr = srs[index]
            stride = stride_in_seconds * sr
            assert isinstance(stride, int), "stride must be an integer"
            output = output[:stride]  # crop to stride
            output_dict[index].append(output)

    # Concatenate and crop the outputs for each layer to match the original waveform length.
    outputs = []
    for index, output in enumerate(output_dict.values()):
        output = torch.cat(output, dim=0)
        sr = srs[index]
        samples = int(wav_length_in_seconds * sr)
        output = output[:samples]
        outputs.append(output)

    return outputs


if __name__ == "__main__":
    from transformers import AutoModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593").to(
        device
    )
    waveform = torch.randn(163840)
    sample_rate = 16000
    outputs = extract_AST(device, model, waveform, sample_rate)
    print("\033[91m" + "outputs: " + "\033[92m", outputs)
