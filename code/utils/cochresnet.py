import importlib.util
import os
import torch
from torch import nn
import torchaudio
from torch.amp import autocast
import chcochleagram
from robustness.tools.helpers import InputNormalize, AudioInputRepresentation
from robustness.audio_models.custom_modules import SequentialWithArgs
import numpy as np
import hdf5storage
from general_analysis_code.preprocess import align_time
from robustness.audio_models.resnet_multi_task import conv1x1, BasicBlock, Bottleneck
from robustness.audio_models.resnet import model_urls
import torch.utils.model_zoo as model_zoo
from utils.pc import (
    generate_pca_pipeline,
    apply_pca_pipeline,
    generate_pca_pipeline_from_weights,
)
from utils.shared import write_summary, prepare_waveform


def cochresnet(
    device,
    output_root,
    stim_names,
    wav_dir,
    out_sr=100,
    pc=100,
    time_window=[-1, 1],
    pca_weights_from=None,
    compute_original=True,
    meta_only=False,
    **kwargs,
):
    half = kwargs.get("half", False)
    feature_name = "cochresnet"
    variant = "wsa"

    if compute_original:
        CochResNet_model = build_model()
        for stim_index, stim_name in enumerate(stim_names):
            wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
            generate_CochResNet_features(
                device=device,
                model=CochResNet_model,
                wav_path=wav_path,
                output_root=output_root,
                n_t=None,
                out_sr=out_sr,
                variant=variant,
                time_window=time_window,
                half=half,
                meta_only=meta_only,
            )

    # compute PC of cochresnet features

    layer_num = 6
    if pc is not None:
        for layer in range(layer_num):
            wav_features = []
            for stim_index, stim_name in enumerate(stim_names):
                wav_path = os.path.join(wav_dir, f"{stim_name}.wav")
                feature_path = f"{output_root}/features/{feature_name}/{variant}_layer{layer}/{stim_name}.mat"
                feature = hdf5storage.loadmat(feature_path)["features"]
                wav_features.append(feature)
            print(f"Start computing PCs for {feature_name} layer {layer}")
            if pca_weights_from is not None:
                weights_path = f"{pca_weights_from}/features/{feature_name}/{variant}_layer{layer}/metadata/pca_weights.mat"
                pca_pipeline = generate_pca_pipeline_from_weights(
                    weights_from=weights_path, pc=pc
                )
            else:
                pca_pipeline = generate_pca_pipeline(
                    wav_features,
                    pc,
                    output_root,
                    feature_name,
                    demean=True,
                    std=False,
                    variant=f"{variant}_layer{layer}",
                )
            feature_variant_out_dir = apply_pca_pipeline(
                wav_features,
                pc,
                output_root,
                feature_name,
                stim_names,
                variant=f"{variant}_layer{layer}",
                pca_pipeline=pca_pipeline,
                time_window=time_window,
                sampling_rate=out_sr,
                meta_only=meta_only,
            )


def generate_CochResNet_features(
    wav_path,
    model,
    output_root,
    n_t=None,
    out_sr=100,
    device="cuda",
    variant="",
    time_window=[-1, 1],
    half=False,
    meta_only=False,
):
    feature = "cochresnet"
    variants = [f"{variant}_layer{i}" for i in range(6)]
    (
        wav_name_no_ext,
        waveform,
        sample_rate,
        t_num_new,
        t_new,
        feature_variant_out_dirs,
    ) = prepare_waveform(
        out_sr, wav_path, output_root, n_t, time_window, feature, variants
    )
    if not meta_only:
        waveform = torch.as_tensor(waveform)
        t_0s = [0.025] * 6
        srs = [200, 100, 50, 25, 25 / 2, 25 / 4]
        before_pad_number = int(np.max(t_0s) * sample_rate) + 1
        after_pad_number = 1200
        waveform = torch.nn.functional.pad(
            waveform, (before_pad_number, after_pad_number)
        )
        t_0s = np.array(t_0s) - before_pad_number / sample_rate

        if device.type == "mps":
            half = False
        with autocast(device_type=device.type, enabled=half):
            outputs = extract_CochResNet(waveform, model, sample_rate, device=device)

        for i, (feats, t_0, sr) in enumerate(zip(outputs, t_0s, srs)):
            print(f"t_0_{i}={t_0}, sr_{i}={sr}")
            feature_variant_out_dir = feature_variant_out_dirs[i]

            ### align time to start from 0 and make the length to be n_t
            t_length = feats.shape[0]
            t_origin = np.arange(t_length) / sr + t_0

            feats = align_time(feats, t_origin, t_new, "t f")
            print(f"Feature {i}: {feats.shape}")

            # save each layer as mat
            out_mat_path = os.path.join(
                feature_variant_out_dir, f"{wav_name_no_ext}.mat"
            )

            # save data as mat
            hdf5storage.savemat(
                out_mat_path, {"features": feats, "t": t_new + time_window[0]}
            )

            write_summary(
                feature_variant_out_dir,
                time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
                dimensions="[time, feature]",
                sampling_rate=out_sr,
                extra="Nothing",
            )
    else:
        for feature_variant_out_dir in feature_variant_out_dirs:
            write_summary(
                feature_variant_out_dir,
                time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
                dimensions="[time, feature]",
                sampling_rate=out_sr,
                extra="Nothing",
            )


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        if with_latent:
            all_outputs = {}
            all_outputs["input_after_preproc"] = x
        x = self.conv1(x)
        if with_latent:
            all_outputs["conv1"] = x
        x = self.bn1(x)
        if with_latent:
            all_outputs["bn1"] = x
        x = self.relu(x)
        if with_latent:
            all_outputs["conv1_relu1"] = x
        x = self.maxpool(x)
        if with_latent:
            all_outputs["maxpool1"] = x

        x = self.layer1(x)
        if with_latent:
            all_outputs["layer1"] = x
        x = self.layer2(x)
        if with_latent:
            all_outputs["layer2"] = x
        x = self.layer3(x)
        if with_latent:
            all_outputs["layer3"] = x
        x = self.layer4(x, fake_relu=fake_relu, no_relu=no_relu)
        if with_latent:
            all_outputs["layer4"] = x

        x = self.avgpool(x)
        if with_latent:
            all_outputs["avgpool"] = x

        return all_outputs


def resnet_multi_task50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls["resnet_multi_task50"]))
    return model


def hack_model(waveform, backbone, normalize, cochleagram_1):
    device = waveform.device
    n_sample = waveform.shape[0]
    cochleagram_1["rep_kwargs"]["signal_size"] = n_sample
    rep_layer = AudioInputRepresentation(**cochleagram_1).to(device)
    with torch.no_grad():
        all_outputs_hack = backbone(rep_layer(normalize(waveform)), with_latent=True)
    return all_outputs_hack


def build_model(
    model_dir="/scratch/snormanh_lab/shared/code/cochdnn/model_directories/resnet50_word_speaker_audioset",
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
    model = resnet_multi_task50()
    model.load_state_dict(sd, strict=False)
    model.eval()
    return model


def extract_CochResNet(waveform, model, sample_rate, device="cuda"):
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
        "conv1_relu1",
        "layer1",
        "layer2",
        "layer3",
        "layer4",
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


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model().to("cuda")
    project = "intracranial-natsound165"
    output_root = os.path.abspath(
        f"{__file__}/../../../projects_toy/{project}/analysis"
    )
    wav_dir = os.path.abspath(
        f"{__file__}/../../../projects_toy/intracranial-natsound165/stimuli/stimulus_audio"
    )
    stim_names = ["stim5_alarm_clock"]
    cochresnet(device, output_root, stim_names, wav_dir, out_sr=100, pc=100)
