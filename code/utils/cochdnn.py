import importlib.util
import os
import torch
from torch import nn
import torchaudio
import chcochleagram
from robustness.tools.helpers import InputNormalize, AudioInputRepresentation
from robustness.audio_models.layers import conv2d_same, pool2d_same
from robustness.audio_models.custom_modules import FakeReLUM
import numpy as np
import hdf5storage
from general_analysis_code.preprocess import align_time
from utils.pc import generate_pc


class AuditoryCNNMultiTask(nn.Module):
    def __init__(self, num_classes=1000):
        super(AuditoryCNNMultiTask, self).__init__()

        # Include initial batch norm to center the cochleagrams
        self.batchnorm0 = nn.BatchNorm2d(1)
        # Initialize this input batch norm to have some reasonable values for the cochleagram
        # Values were extracted from a word network that was trained and initialized in the typical way
        # These values help with the audioset training.
        nn.init.constant_(self.batchnorm0.weight, 0.6180)
        nn.init.constant_(self.batchnorm0.bias, -2.0971)

        self.conv0 = conv2d_same.create_conv2d_pad(
            1, 96, kernel_size=[7, 14], stride=[3, 3], padding="same"
        )
        self.relu0 = nn.ReLU(inplace=False)
        self.maxpool0 = pool2d_same.create_pool2d(
            "max", kernel_size=[2, 5], stride=[2, 2], padding="same"
        )
        self.batchnorm1 = nn.BatchNorm2d(96)

        self.conv1 = conv2d_same.create_conv2d_pad(
            96, 256, kernel_size=[4, 8], stride=[2, 2], padding="same"
        )
        self.relu1 = nn.ReLU(inplace=False)
        self.maxpool1 = pool2d_same.create_pool2d(
            "max", kernel_size=[2, 5], stride=[2, 2], padding="same"
        )
        self.batchnorm2 = nn.BatchNorm2d(256)

        self.conv2 = conv2d_same.create_conv2d_pad(
            256, 512, kernel_size=[2, 5], stride=[1, 1], padding="same"
        )
        self.relu2 = nn.ReLU(inplace=False)

        self.conv3 = conv2d_same.create_conv2d_pad(
            512, 1024, kernel_size=[2, 5], stride=[1, 1], padding="same"
        )
        self.relu3 = nn.ReLU(inplace=False)

        self.conv4 = conv2d_same.create_conv2d_pad(
            1024, 512, kernel_size=[2, 5], stride=[1, 1], padding="same"
        )
        self.relu4 = nn.ReLU(inplace=False)

        self.avgpool = pool2d_same.create_pool2d(
            "avg", kernel_size=[2, 5], stride=[2, 2], padding="same"
        )

        self.fake_relu_dict = nn.ModuleDict()
        self.fake_relu_dict["relu0"] = FakeReLUM()
        self.fake_relu_dict["relu1"] = FakeReLUM()
        self.fake_relu_dict["relu2"] = FakeReLUM()
        self.fake_relu_dict["relu3"] = FakeReLUM()
        self.fake_relu_dict["relu4"] = FakeReLUM()

    def forward(self, x, fake_relu=False, with_latent=True):
        all_outputs = {}
        all_outputs["input_after_preproc"] = x

        x = self.batchnorm0(x)
        all_outputs["batchnorm0"] = x

        x = self.conv0(x)
        all_outputs["conv0"] = x

        if fake_relu:
            all_outputs["relu0" + "_fakerelu"] = self.fake_relu_dict["relu0"](x)

        x = self.relu0(x)
        all_outputs["relu0"] = x

        x = self.maxpool0(x)
        all_outputs["maxpool0"] = x

        x = self.batchnorm1(x)
        all_outputs["batchnorm1"] = x

        x = self.conv1(x)
        all_outputs["conv1"] = x

        if fake_relu:
            all_outputs["relu1" + "_fakerelu"] = self.fake_relu_dict["relu1"](x)

        x = self.relu1(x)
        all_outputs["relu1"] = x

        x = self.maxpool1(x)
        all_outputs["maxpool1"] = x

        x = self.batchnorm2(x)
        all_outputs["batchnorm2"] = x

        x = self.conv2(x)
        all_outputs["conv2"] = x

        if fake_relu:
            all_outputs["relu2" + "_fakerelu"] = self.fake_relu_dict["relu2"](x)

        x = self.relu2(x)
        all_outputs["relu2"] = x

        x = self.conv3(x)
        all_outputs["conv3"] = x

        if fake_relu:
            all_outputs["relu3" + "_fakerelu"] = self.fake_relu_dict["relu3"](x)

        x = self.relu3(x)
        all_outputs["relu3"] = x

        x = self.conv4(x)
        all_outputs["conv4"] = x

        if fake_relu:
            all_outputs["relu4" + "_fakerelu"] = self.fake_relu_dict["relu4"](x)

        x = self.relu4(x)
        all_outputs["relu4"] = x

        x = self.avgpool(x)
        all_outputs["avgpool"] = x
        return all_outputs


def hack_model(waveform, backbone, normalize, cochleagram_1):
    device = waveform.device
    n_sample = waveform.shape[0]
    cochleagram_1["rep_kwargs"]["signal_size"] = n_sample
    rep_layer = AudioInputRepresentation(**cochleagram_1).to(device)
    with torch.no_grad():
        all_outputs_hack = backbone(rep_layer(normalize(waveform)), with_latent=True)
    return all_outputs_hack


def build_model(
    model_dir="/home/gliao2/snormanh_lab_shared/code/cochdnn/model_directories/kell2018_word_speaker_audioset",
):
    build_network_spec = importlib.util.spec_from_file_location(
        "build_network", os.path.join(model_dir, "build_network.py")
    )
    build_network = importlib.util.module_from_spec(build_network_spec)
    build_network_spec.loader.exec_module(build_network)
    model1, _ = build_network.main(include_rep_in_model=True)
    sd = model1.model.state_dict()
    # remove the 0. from the beginning of the key
    sd = {k[2:]: v for k, v in sd.items()}
    # remove first 3 items
    sd = {k: v for k, v in sd.items() if not k.startswith("full_rep")}
    del model1
    model = AuditoryCNNMultiTask()
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def extract_CochDNN(waveform, model, sample_rate, device="cuda"):
    normalize = InputNormalize(torch.tensor([0]), torch.tensor([1]), -1, 1).cuda()
    cochleagram_1 = {
        "rep_type": "cochleagram",
        "rep_kwargs": {
            "signal_size": 40000,
            "sr": 20000,
            "env_sr": 200,
            "pad_factor": None,
            "use_rfft": True,
            "coch_filter_type": chcochleagram.cochlear_filters.ERBCosFilters,
            "coch_filter_kwargs": {
                "n": 50,
                "low_lim": 50,
                "high_lim": 10000,
                "sample_factor": 4,
                "full_filter": False,
            },
            "env_extraction_type": chcochleagram.envelope_extraction.HilbertEnvelopeExtraction,
            "downsampling_type": chcochleagram.downsampling.SincWithKaiserWindow,
            "downsampling_kwargs": {"window_size": 1001},
        },
        "compression_type": "coch_p3",
        "compression_kwargs": {
            "scale": 1,
            "offset": 1e-8,
            "clip_value": 5,  # This wil clip cochleagram values < ~0.04
            "power": 0.3,
        },
    }
    model_input_sample_rate = cochleagram_1["rep_kwargs"]["sr"]
    # Convert waveform to tensor if it's not already.
    waveform = torch.as_tensor(waveform).reshape(-1)

    # Resample the waveform to the target sample rate if necessary.
    if sample_rate != model_input_sample_rate:
        waveform = torchaudio.functional.resample(
            waveform, sample_rate, model_input_sample_rate
        )
    waveform = waveform.to(device)
    model = model.to(device)
    normalize = normalize.to(device)
    all_outputs_hack = hack_model(waveform, model, normalize, cochleagram_1)
    activation_keys = [
        "input_after_preproc",
        "relu0",
        "relu1",
        "relu2",
        "relu3",
        "relu4",
    ]
    activations = [
        all_outputs_hack[key]
        .reshape(-1, all_outputs_hack[key].shape[3])
        .T.detach()
        .cpu()
        .numpy()
        for key in activation_keys
    ]
    return activations


def generate_CochDNN_features(
    wav_path, model, output_root, n_t=None, out_sr=100, device="cuda", variant=""
):
    wav_name = os.path.basename(wav_path)
    wav_name_no_ext = os.path.splitext(wav_name)[0]
    feature_class_out_dir = os.path.join(output_root, "features", "cochdnn")
    meta_out_path = os.path.join(output_root, "feature_metadata", "cochdnn.txt")
    if not os.path.exists(feature_class_out_dir):
        os.makedirs(feature_class_out_dir)
    if not os.path.exists(os.path.dirname(meta_out_path)):
        os.makedirs(os.path.dirname(meta_out_path))
    waveform, sample_rate = torchaudio.load(wav_path)
    waveform = waveform.reshape(-1)
    t_0s = [0.025, 0.0325, 0.0475, 0.0475, 0.0475, 0.0475]
    srs = [200, 200 / 3, 50 / 3, 25 / 3, 25 / 3, 25 / 3]
    before_pad_number = int(np.max(t_0s) * sample_rate) + 1
    after_pad_number = 1200
    t_num_new = int(np.round(waveform.shape[0] / sample_rate * out_sr))
    waveform = torch.nn.functional.pad(waveform, (before_pad_number, after_pad_number))
    t_0s = np.array(t_0s) - before_pad_number / sample_rate

    if n_t is not None:
        t_new = np.arange(n_t) / out_sr
    else:
        t_new = np.arange(t_num_new) / out_sr
    outputs = extract_CochDNN(waveform, model, sample_rate, device=device)

    for i, (feats, t_0, sr) in enumerate(zip(outputs, t_0s, srs)):
        print(f"t_0_{i}={t_0}, sr_{i}={sr}")
        feature_variant_out_dir = os.path.join(
            feature_class_out_dir, f"{variant}_layer{i}"
        )
        if not os.path.exists(feature_variant_out_dir):
            os.makedirs(feature_variant_out_dir)

        ### align time to start from 0 and make the length to be n_t
        t_length = feats.shape[0]
        t_origin = np.arange(t_length) / sr + t_0

        feats = align_time(feats, t_origin, t_new, "t f")
        print(f"Feature {i}: {feats.shape}")

        # save each layer as mat
        out_mat_path = os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.mat")

        # save data as mat
        hdf5storage.savemat(out_mat_path, {"features": feats})
        # generate meta file for this layer
        with open(
            os.path.join(feature_variant_out_dir, f"{wav_name_no_ext}.txt"), "w"
        ) as f:
            f.write(f"""The shape of this layer is {feats.shape}.""")

    # generate meta file for this feature
    with open(meta_out_path, "w") as f:
        f.write(
            f"""CochDNN features.
Each layer is saved separately as variant. 
Timestamps start from 0 ms, sr={out_sr}Hz. You can find out the shape of each stimulus in features/cochdnn/<layer>/<stimuli>.txt as (n_time,n_feature)."""
        )


def cochdnn(device, output_root, stim_names, wav_dir, out_sr=100, pc=100, **kwargs):
    CochDNN_model = build_model()
    variant = "wsa"
    for stim_index, stim_name in enumerate(stim_names):
        wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
        generate_CochDNN_features(
            device=device,
            model=CochDNN_model,
            wav_path=wav_path,
            output_root=output_root,
            n_t=None,
            out_sr=out_sr,
            variant=variant,
        )

    # compute PC of cochdnn features
    feature_name = "cochdnn"
    layer_num = 6
    for layer in range(layer_num):
        wav_features = []
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            feature_path = f"{output_root}/features/{feature_name}/{variant}_layer{layer}/{stim_name}.mat"
            feature = hdf5storage.loadmat(feature_path)["features"]
            wav_features.append(feature)
        pca_pipeline = generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            demean=True,
            std=False,
            variant=f"{variant}_layer{layer}",
        )
        generate_pc(
            wav_features,
            pc,
            output_root,
            feature_name,
            stim_names,
            demean=True,
            std=False,
            variant=f"{variant}_layer{layer}",
            pca_pipeline=pca_pipeline,
        )


if __name__ == "__main__":
    sample_rate = 20000
    waveform = torch.randn(sample_rate * 2).to("cuda")
    model = build_model().to("cuda")
    generate_CochDNN_features(
        "/scratch/snormanh_lab/shared/projects/intracranial-natsound119/stimuli/stimulus_audio/aninonvoc_cat83_rec1_cicadas_excerpt1.wav",
        model,
        "/scratch/snormanh_lab/shared/Sigurd/projects/intracranial-natsound165/analysis/",
    )
    pass
