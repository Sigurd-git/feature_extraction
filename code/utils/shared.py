import os
import torchaudio
import numpy as np
import hdf5storage
import matlab
import time
import torch
import pytest


def write_summary(
    feature_variant_out_dir,
    time_window="1 second before to 1 second after",
    dimensions="[time, feature]",
    sampling_rate=100,
    extra="Nothing",
    url="https://github.com/Sigurd-git/feature_extraction/tree/main/code/utils",
    parameter_dict=None,
):
    meta_out_dir = os.path.join(feature_variant_out_dir, "metadata")
    os.makedirs(meta_out_dir, exist_ok=True)

    meta_out_path = os.path.join(meta_out_dir, "summary.txt")

    # get yyyy-mm-dd-hh:mm:ss
    now = time.strftime("%Y-%m-%d-%H:%M:%S")

    # generate meta file for this feature
    with open(meta_out_path, "w") as f:
        f.write(
            f"""Created on: {now};
Time window: {time_window};
The dimensions of the matrices: {dimensions};
The sampling rate of the features: {sampling_rate} Hz;
Code used to generate those features are here: {url};
Extra comments: {extra}."""
        )
    if parameter_dict is not None:
        parameter_out_path = os.path.join(meta_out_dir, "parameters.mat")
        for key, value in parameter_dict.items():
            if isinstance(value, matlab.double):
                print(f"Converting {key} to numpy array")
                parameter_dict[key] = np.array(value._data)
        hdf5storage.savemat(parameter_out_path, parameter_dict)


def prepare_waveform(
    out_sr, wav_path, output_root, n_t, time_window, feature, variants
):
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    print(f"Getting {feature} features for stim: {wav_path}")
    feature_variant_out_dirs = [
        create_feature_variant_dir(output_root, feature, variant)[1]
        for variant in variants
    ]

    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.reshape(-1)
    # pad waveform according to time_window
    assert time_window[0] <= 0 and time_window[1] >= 0
    before_pad = int(abs(time_window[0]) * sample_rate)
    after_pad = int(abs(time_window[1]) * sample_rate)
    waveform = np.pad(waveform, (before_pad, after_pad), "constant")

    t_num_new = int(np.round(waveform.shape[0] / sample_rate * out_sr))
    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr
    t_num_new = len(t_new)

    return (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    )


def create_feature_variant_dir(output_root, feature, variant):
    feature_class_out_dir = os.path.join(output_root, "features", feature)
    feature_variant_out_dir = os.path.join(feature_class_out_dir, variant)
    os.makedirs(feature_variant_out_dir, exist_ok=True)
    return feature_class_out_dir, feature_variant_out_dir


def pad_split_waveform(
    waveform, kernel=130, overlap=10, sr=100, fix_length=False, out_sr=1
):
    stride = kernel - overlap

    new_stride = np.round(stride * out_sr) / out_sr
    new_kernel = int(kernel * sr) / sr

    kernel_in_sample = int(new_kernel * sr)
    stride_in_sample = new_stride * sr

    assert np.isclose(
        stride_in_sample - np.round(stride_in_sample), 0
    ), "stride_in_sample must be an integer"

    stride_in_sample = int(np.round(stride_in_sample))

    stride_in_output_sample = stride_in_sample / sr * out_sr
    assert np.isclose(
        stride_in_output_sample - np.round(stride_in_output_sample), 0
    ), "stride_in_output_sample must be an integer"
    stride_in_output_sample = int(np.round(stride_in_output_sample))

    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    assert len(waveform.shape) == 1, "waveform should be 1D tensor"

    # if waveform is less than kernel seconds, then return waveform directly.
    if len(waveform) <= kernel_in_sample:
        if fix_length:
            # pad the waveform
            waveform = torch.nn.functional.pad(
                waveform, (0, kernel_in_sample - len(waveform))
            )
        splitted_waveforms = torch.stack([waveform])
    else:
        # pad waveform to kernel_in_sample+ n*stride_in_sample
        n_stride = torch.ceil(
            torch.tensor((len(waveform) - kernel_in_sample) / stride_in_sample)
        )
        desired_length = int(kernel_in_sample + n_stride * stride_in_sample)
        n_window = int(n_stride + 1)
        waveform = torch.nn.functional.pad(
            waveform, (0, desired_length - len(waveform))
        )
        waveform = waveform.reshape(1, 1, 1, -1)
        splitted_waveforms = torch.nn.functional.unfold(
            waveform, kernel_size=(1, kernel_in_sample), stride=(1, stride_in_sample)
        )  # 1,kernel_size,n_window
        splitted_waveforms = splitted_waveforms.squeeze(0).T  # n_window,kernel_size
        assert n_window == splitted_waveforms.shape[0]
    return splitted_waveforms, stride_in_output_sample


def concatenate_with_overlap(outputs, stride_in_output_sample):
    # Crop and append the outputs based on the stride for each layer.
    assert isinstance(stride_in_output_sample, int), "stride must be an integer"
    n_window = len(outputs)
    length_concatenated_output = (
        len(outputs[0]) + (n_window - 1) * stride_in_output_sample
    )
    concatenated_output = torch.zeros(
        length_concatenated_output, outputs[0].shape[1], 2
    )

    assert 2 * stride_in_output_sample >= len(
        outputs[0]
    ), "stride is too small, a frame should not overlap with more than 2 frames."

    for index, output in enumerate(outputs):
        first = output[:stride_in_output_sample]  # crop to stride
        rest = output[stride_in_output_sample:]
        n_rest = len(rest)
        concatenated_output[
            index * stride_in_output_sample : (index + 1) * stride_in_output_sample,
            :,
            0,
        ] = first
        concatenated_output[
            (index + 1) * stride_in_output_sample : (index + 1)
            * stride_in_output_sample
            + n_rest,
            :,
            1,
        ] = rest
    if n_rest == 1:
        weights = torch.ones((1, 2))
    else:
        weights = np.hanning(n_rest * 2).reshape(-1, 2)
        weights = torch.tensor(weights)

    # normalize to 1
    weights = weights / torch.sum(weights, axis=1, keepdims=True)
    weights = weights.T

    weights_first_part = torch.concatenate(
        [weights[0], torch.ones(stride_in_output_sample - n_rest)]
    )
    weights_second_part = torch.concatenate(
        [weights[1], torch.zeros(stride_in_output_sample - n_rest)]
    )
    weights_0 = torch.concatenate(
        [torch.ones(stride_in_output_sample)]
        + [weights_first_part] * (n_window - 1)
        + [torch.zeros(n_rest)]
    )
    weights_1 = torch.concatenate(
        [torch.zeros(stride_in_output_sample)]
        + [weights_second_part] * (n_window - 1)
        + [torch.ones(n_rest)]
    )

    assert np.allclose(
        weights_0 + weights_1, 1
    ), "weights_0 and weights_1 should sum to 1"
    weights_0 = torch.tensor(weights_0).reshape(-1, 1)
    weights_1 = torch.tensor(weights_1).reshape(-1, 1)
    concatenated_output[:, :, 0] = concatenated_output[:, :, 0] * weights_0
    concatenated_output[:, :, 1] = concatenated_output[:, :, 1] * weights_1
    concatenated_output = torch.sum(concatenated_output, 2)

    return concatenated_output


def test_extract_features(x, downsample=4):
    x = torch.tensor(x)  # time

    x = x.reshape(1, 1, 1, -1)
    x = torch.nn.functional.unfold(
        x, kernel_size=(1, downsample), stride=(1, downsample)
    )[0]  # downsample,n_window
    x = torch.mean(x, 0)  # n_window
    x = torch.stack([x, x, x], 1)  # n_window,3
    return x


# Returns a list with one element if waveform is less than kernel seconds.
def test_waveform_less_than_kernel():
    waveform = [1, 2, 3, 4, 5]
    kernel = 10
    overlap = 2
    sr = 100
    fix_length = False

    result = pad_split_waveform(waveform, kernel, overlap, sr, fix_length)

    assert np.allclose(result[0][0].tolist(), [1, 2, 3, 4, 5])


def test_waveform_not_1D_tensor():
    waveform = torch.tensor([[1, 2, 3], [4, 5, 6]])
    kernel = 10
    overlap = 2
    sr = 100
    fix_length = False

    with pytest.raises(AssertionError):
        pad_split_waveform(waveform, kernel, overlap, sr, fix_length)


def test_waveform_greater_than_kernel():
    waveform = np.float32(np.arange(100) + 1)
    kernel = 5
    overlap = 4  # overlap/sr*out_sr >=1
    sr = 2
    fix_length = False

    splitted_waveforms, stride_in_output_sample = pad_split_waveform(
        waveform, kernel, overlap, sr, fix_length, out_sr=0.5
    )
    assert np.allclose(splitted_waveforms[0].tolist(), waveform[: kernel * sr])

    gd = test_extract_features(waveform)
    extracted_features = [
        test_extract_features(splitted_waveform)
        for splitted_waveform in splitted_waveforms
    ]
    concatenated_output = concatenate_with_overlap(
        extracted_features, stride_in_output_sample
    )
    assert np.allclose(concatenated_output, gd)


def test_waveform_like_real():
    sr = 2000
    seconds = 20
    waveform = np.float32(np.arange(seconds * sr) + 1)
    kernel = 5
    overlap = 1  # overlap/sr*out_sr >=1

    fix_length = False

    splitted_waveforms, stride_in_output_sample = pad_split_waveform(
        waveform, kernel, overlap, sr, fix_length, out_sr=sr / 4
    )
    assert np.allclose(splitted_waveforms[0].tolist(), waveform[: kernel * sr])

    gd = test_extract_features(waveform)
    extracted_features = [
        test_extract_features(splitted_waveform)
        for splitted_waveform in splitted_waveforms
    ]
    concatenated_output = concatenate_with_overlap(
        extracted_features, stride_in_output_sample
    )
    assert np.allclose(concatenated_output[: len(gd)], gd)


if __name__ == "__main__":
    test_waveform_less_than_kernel()
    test_waveform_not_1D_tensor()
    test_waveform_greater_than_kernel()
    test_waveform_like_real()
    print("All tests passed!")
