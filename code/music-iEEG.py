# import torch
import os
import glob
import pandas as pd
import numpy as np
from utils.feature_extraction import generate_notes_features

######personalize parameters

output_root = "/scratch/snormanh_lab/shared/projects/music-iEEG/analysis"
wav_dir = "/scratch/snormanh_lab/shared/projects/music-iEEG/stimuli/stimulus_audio"
wav_paths = glob.glob(f"{wav_dir}/*.wav")
note_timing_csvs = glob.glob(
    "/home/gliao2/snormanh_lab_shared/projects/music-iEEG/analysis/origin_notes/*/*/*/*/*.csv"
)

notes_labels, bin_pitch_labels, rela_pitch_labels = [], [], []
for csv_path in note_timing_csvs:
    # read in the csv file
    df_csv = pd.read_csv(csv_path)
    notes_labels.extend(df_csv["pitch"].values)
    bin_pitch_labels.extend(df_csv["bin_pitch"].values)
    rela_pitch_labels.extend(df_csv["key_relative_pitch"].values)

all_notes_labels = np.unique(notes_labels)
all_bin_pitch_labels = np.unique(bin_pitch_labels)
all_rela_pitch_labels = np.unique(rela_pitch_labels)
# remove nan
all_notes_labels = all_notes_labels[~np.isnan(all_notes_labels)].astype(int)
all_bin_pitch_labels = all_bin_pitch_labels[~np.isnan(all_bin_pitch_labels)].astype(int)
all_rela_pitch_labels = all_rela_pitch_labels[~np.isnan(all_rela_pitch_labels)].astype(
    int
)
n_t = 6000

# read in /scratch/snormanh_lab/shared/projects/music-iEEG/stimuli/stimulus_metadata.txt, from second line, then this is a csv file

file_path = (
    "/scratch/snormanh_lab/shared/projects/music-iEEG/stimuli/stimulus_metadata.txt"
)
df = pd.read_csv(file_path, header=1)


for wav_path in wav_paths:
    wav_name = os.path.basename(wav_path)
    # get the corresponding row in the csv file
    row = df.loc[df["Target_path"] == wav_name]
    assert len(row) == 1
    # get the corresponding Origin_path
    origin_wav_path = row["Origin_path"].values[0]
    # replace .wav with .csv
    origin_csv_path = origin_wav_path.replace(".wav", ".csv")
    # replace origin_stimuli with origin_notes
    origin_csv_path = origin_csv_path.replace("origin_stimuli", "origin_notes")

    # name of the csv file
    csv_name = os.path.splitext(os.path.basename(origin_csv_path))[0]
    # split by _
    csv_name_split = csv_name.split("_")
    csv_dir = os.path.dirname(origin_csv_path)
    split_ = csv_name_split[0]
    Track = csv_name_split[1]
    csv_name = csv_name.replace("-", "_")
    csv_path = os.path.join(csv_dir, split_, Track, "all_src", f"{csv_name}.csv")

    sr = 100
    if os.path.exists(csv_path):
        # read in the csv file
        df_csv = pd.read_csv(csv_path)

        # bin_pitch onset/duration feature
        bin_pitch_onsets = df_csv["onset"].values
        bin_pitch_offsets = df_csv["offset"].values
        bin_pitch_labels = df_csv["bin_pitch"].values
        # remove nan
        bin_pitch_onsets = bin_pitch_onsets[~np.isnan(bin_pitch_labels)]
        bin_pitch_offsets = bin_pitch_offsets[~np.isnan(bin_pitch_labels)]
        bin_pitch_labels = bin_pitch_labels[~np.isnan(bin_pitch_labels)].astype(int)

        generate_notes_features(
            output_root,
            wav_path,
            bin_pitch_labels,
            bin_pitch_onsets,
            bin_pitch_offsets,
            n_t,
            all_bin_pitch_labels,
            onset_feature=True,
            appendix="_bin_pitch_onset",
            discrete=False,
        )

        generate_notes_features(
            output_root,
            wav_path,
            bin_pitch_labels,
            bin_pitch_onsets,
            bin_pitch_offsets,
            n_t,
            all_bin_pitch_labels,
            appendix="_bin_pitch_duration",
            discrete=False,
        )

        # rela_pitch onset feature
        rela_pitch_onsets = df_csv["onset"].values
        rela_pitch_offsets = df_csv["offset"].values
        rela_pitch_labels = df_csv["key_relative_pitch"].values
        # remove nan
        rela_pitch_onsets = rela_pitch_onsets[~np.isnan(rela_pitch_labels)]
        rela_pitch_offsets = rela_pitch_offsets[~np.isnan(rela_pitch_labels)]
        rela_pitch_labels = rela_pitch_labels[~np.isnan(rela_pitch_labels)].astype(int)

        generate_notes_features(
            output_root,
            wav_path,
            rela_pitch_labels,
            rela_pitch_onsets,
            rela_pitch_offsets,
            n_t,
            all_rela_pitch_labels,
            onset_feature=True,
            appendix="_rela_pitch_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            rela_pitch_labels,
            rela_pitch_onsets,
            rela_pitch_offsets,
            n_t,
            all_rela_pitch_labels,
            appendix="_rela_pitch_duration",
            discrete=False,
        )

        # Note onset feature
        notes_onsets = df_csv["onset"].values
        notes_offsets = df_csv["offset"].values
        # remove nan
        notes_onsets = notes_onsets[~np.isnan(notes_onsets)]
        notes_offsets = notes_offsets[~np.isnan(notes_offsets)]
        notes_labels = ["all"] * len(notes_onsets)
        all_notes_labels = ["all"]
        generate_notes_features(
            output_root,
            wav_path,
            notes_labels,
            notes_onsets,
            notes_offsets,
            n_t,
            all_notes_labels,
            onset_feature=True,
            appendix="_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            notes_labels,
            notes_onsets,
            notes_offsets,
            n_t,
            all_notes_labels,
            discrete=False,
            appendix="_duration",
        )

        # One-hot code the beat
        beat_onsets = df_csv["beats"].values
        beat_onsets = beat_onsets[~np.isnan(beat_onsets)]
        beat_offsets = beat_onsets + 2 / sr
        beat_labels = ["beat"] * len(beat_onsets)
        all_beat_labels = ["beat"]
        generate_notes_features(
            output_root,
            wav_path,
            beat_labels,
            beat_onsets,
            beat_offsets,
            n_t,
            all_beat_labels,
            onset_feature=True,
            appendix="_beat_onset",
            discrete=False,
        )

        # One-hot code the downbeat
        downbeat_onsets = df_csv["downbeats"].values
        downbeat_onsets = downbeat_onsets[~np.isnan(downbeat_onsets)]
        downbeat_offsets = downbeat_onsets + 2 / sr
        downbeat_labels = ["downbeat"] * len(downbeat_onsets)
        all_downbeat_labels = ["downbeat"]
        generate_notes_features(
            output_root,
            wav_path,
            downbeat_labels,
            downbeat_onsets,
            downbeat_offsets,
            n_t,
            all_downbeat_labels,
            onset_feature=True,
            appendix="_downbeat_onset",
            discrete=False,
        )

        # Drum set onset
        drum_set_onsets = df_csv["onset"].values
        drum_set_offsets = df_csv["offset"].values
        drum_set_masks = df_csv["is_drum"].values.astype(bool)
        drum_set_onsets = drum_set_onsets[drum_set_masks]
        drum_set_offsets = drum_set_offsets[drum_set_masks]
        drum_labels = ["drum"] * len(drum_set_onsets)
        all_drum_labels = ["drum"]
        generate_notes_features(
            output_root,
            wav_path,
            drum_labels,
            drum_set_onsets,
            drum_set_offsets,
            n_t,
            all_drum_labels,
            onset_feature=True,
            appendix="_drum_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            drum_labels,
            drum_set_onsets,
            drum_set_offsets,
            n_t,
            all_drum_labels,
            appendix="_drum_duration",
            discrete=False,
        )

        # Piano onset
        piano_onsets = df_csv["onset"].values
        piano_offsets = df_csv["offset"].values
        piano_masks = df_csv["is_piano"].values.astype(bool)
        piano_onsets = piano_onsets[piano_masks]
        piano_offsets = piano_offsets[piano_masks]
        piano_labels = ["piano"] * len(piano_onsets)
        all_piano_labels = ["piano"]
        generate_notes_features(
            output_root,
            wav_path,
            piano_labels,
            piano_onsets,
            piano_offsets,
            n_t,
            all_piano_labels,
            onset_feature=True,
            appendix="_piano_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            piano_labels,
            piano_onsets,
            piano_offsets,
            n_t,
            all_piano_labels,
            appendix="_piano_duration",
            discrete=False,
        )

        # Guitar onset
        guitar_onsets = df_csv["onset"].values
        guitar_offsets = df_csv["offset"].values
        guitar_masks = df_csv["is_guitar"].values.astype(bool)
        guitar_onsets = guitar_onsets[guitar_masks]
        guitar_offsets = guitar_offsets[guitar_masks]
        guitar_labels = ["guitar"] * len(guitar_onsets)
        all_guitar_labels = ["guitar"]
        generate_notes_features(
            output_root,
            wav_path,
            guitar_labels,
            guitar_onsets,
            guitar_offsets,
            n_t,
            all_guitar_labels,
            onset_feature=True,
            appendix="_guitar_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            guitar_labels,
            guitar_onsets,
            guitar_offsets,
            n_t,
            all_guitar_labels,
            appendix="_guitar_duration",
            discrete=False,
        )

        # is_beat onset
        is_beat_onsets = df_csv["onset"].values
        is_beat_offsets = df_csv["offset"].values
        is_beat_masks = df_csv["is_beat"].values.astype(bool)
        is_beat_onsets = is_beat_onsets[is_beat_masks]
        is_beat_offsets = is_beat_offsets[is_beat_masks]
        is_beat_labels = ["is_beat"] * len(is_beat_onsets)
        all_is_beat_labels = ["is_beat"]
        generate_notes_features(
            output_root,
            wav_path,
            is_beat_labels,
            is_beat_onsets,
            is_beat_offsets,
            n_t,
            all_is_beat_labels,
            onset_feature=True,
            appendix="_is_beat_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            is_beat_labels,
            is_beat_onsets,
            is_beat_offsets,
            n_t,
            all_is_beat_labels,
            appendix="_is_beat_duration",
            discrete=False,
        )

        # is_downbeat onset
        is_downbeat_onsets = df_csv["onset"].values
        is_downbeat_offsets = df_csv["offset"].values
        is_downbeat_masks = df_csv["is_downbeat"].values.astype(bool)
        is_downbeat_onsets = is_downbeat_onsets[is_downbeat_masks]
        is_downbeat_offsets = is_downbeat_offsets[is_downbeat_masks]
        is_downbeat_labels = ["is_downbeat"] * len(is_downbeat_onsets)
        all_is_downbeat_labels = ["is_downbeat"]
        generate_notes_features(
            output_root,
            wav_path,
            is_downbeat_labels,
            is_downbeat_onsets,
            is_downbeat_offsets,
            n_t,
            all_is_downbeat_labels,
            onset_feature=True,
            appendix="_is_downbeat_onset",
            discrete=False,
        )
        generate_notes_features(
            output_root,
            wav_path,
            is_downbeat_labels,
            is_downbeat_onsets,
            is_downbeat_offsets,
            n_t,
            all_is_downbeat_labels,
            appendix="_is_downbeat_duration",
            discrete=False,
        )
