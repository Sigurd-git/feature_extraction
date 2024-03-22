from torch.linalg import svd
import torch
import numpy as np
import hdf5storage
import os
from utils.shared import write_summary
from gpu_pca import IncrementalPCAonGPU
import uuid


class PCA:
    def __init__(self, n_components, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean = None
        self.components = None
        self.Vt = None

    def fit(self, X, solver="incremental"):
        """
        Learn the PCA weights of dataset X and save U, S, Vt.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """
        X = torch.as_tensor(X, dtype=torch.float32)
        if solver == "svd":
            self.mean = torch.mean(X, dim=0, keepdims=True)
            X = X - self.mean
            _, _, self.Vt = svd(X, full_matrices=False)
            self.components = self.Vt[: self.n_components]
        else:
            gpu_model = IncrementalPCAonGPU(
                n_components=self.n_components, batch_size=5000
            )
            gpu_model.fit(X)
            self.Vt = gpu_model.Vt.cpu()
            self.components = gpu_model.components_.cpu()
            self.mean = gpu_model.mean_.cpu()
        pass

    def load(self, mean, V_all):
        """
        Load the mean, standard deviation, and principal components from saved files to prepare for PCA.

        Args:
            mean (numpy.ndarray): The mean of the data.
            V_all (numpy.ndarray): The principal components of the data.

        Returns:
            None
        """
        self.mean = mean
        self.Vt = V_all.T
        self.components = V_all[:, : self.n_components].T

    def transform(self, X):
        """
        Apply PCA weights to the new data set X and perform SVD decomposition.

        Parameters:

        - X: numpy array of shape (n_samples, n_features)

        Return value:

        - X_transformed: numpy array of shape (n_samples, n_components)

        """
        X = torch.as_tensor(X, dtype=torch.float32)
        # Go to the average
        X = X - self.mean

        # Convert the PCA weight obtained during training
        X_transformed = torch.matmul(X, self.components.T)

        return X_transformed

    def fit_transform(self, X):
        """
        Learn PCA weights and apply them to X, while returning U, S, Vt.

        Parameters:

        - X: numpy array with shape (n_samples, n_features)

        Returns:

        - X_transformed: numpy array with shape (n_samples, n_components)
        """
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed


def generate_pca_pipeline(
    wav_features,
    pc,
    output_root,
    feature_name,
    variant,
    whiten=False,
    appendix="",
):
    # pc should be a number
    # concatenate all features in time
    wav_features = [
        torch.as_tensor(wav_feature, dtype=torch.float32)
        for wav_feature in wav_features
    ]
    feature = torch.concatenate(wav_features, dim=0)

    pca_pipeline = PCA(n_components=pc, whiten=whiten)
    pca_pipeline.fit(feature)

    Vt = pca_pipeline.components
    Vt_all = pca_pipeline.Vt
    # asser output_roots are all the same

    feature_dir = os.path.join(output_root, "features", feature_name)
    variant_dir = os.path.join(feature_dir, f"{variant}{appendix}")

    os.makedirs(variant_dir, exist_ok=True)
    out_mat_path = os.path.join(variant_dir, "metadata", "pca_weights.mat")
    random_filename = str(uuid.uuid4()) + ".mat"
    random_filepath = os.path.join(variant_dir, "metadata", random_filename)
    Vt = np.asarray(Vt)
    Vt_all = np.asarray(Vt_all)
    pca_pipeline_mean = np.asarray(pca_pipeline.mean)
    hdf5storage.savemat(
        random_filepath,
        {
            "V": Vt.T,
            "V_all": Vt_all.T,
            "mean": pca_pipeline_mean,
        },
    )
    os.rename(random_filepath, out_mat_path)
    return pca_pipeline


def apply_pca_pipeline(
    wav_features,
    pc,
    output_root,
    feature_name,
    wav_noext_names,
    pca_pipeline,
    variant=None,
    appendix="",
    time_window=[-1, 1],
    sampling_rate=100,
    meta_only=False,
):
    wav_features = [
        torch.as_tensor(wav_feature, dtype=torch.float32)
        for wav_feature in wav_features
    ]
    feature_dir = os.path.join(output_root, "features", feature_name)

    if (variant is None) or (variant == "") or (variant == "original"):
        variant_dir = os.path.join(feature_dir, f"pc{pc}{appendix}")
    else:
        variant_dir = os.path.join(feature_dir, f"{variant}_pc{pc}{appendix}")

    os.makedirs(os.path.join(variant_dir, "metadata"), exist_ok=True)
    out_weights_path = os.path.join(variant_dir, "metadata", "pca_weights.mat")
    random_filename = str(uuid.uuid4()) + ".mat"
    random_filepath = os.path.join(variant_dir, "metadata", random_filename)
    Vt = pca_pipeline.components
    Vt_all = pca_pipeline.Vt

    Vt = np.asarray(Vt)
    Vt_all = np.asarray(Vt_all)
    pca_pipeline_mean = np.asarray(pca_pipeline.mean)
    hdf5storage.savemat(
        random_filepath,
        {
            "V": Vt.T,
            "V_all": Vt_all.T,
            "mean": pca_pipeline_mean,
        },
    )
    os.rename(random_filepath, out_weights_path)
    if not meta_only:
        feature = torch.concatenate(wav_features, dim=0)
        features_pc = pca_pipeline.transform(feature)
        # split back to each wav
        nts = [len(wav_feature) for wav_feature in wav_features]
        wav_features_pc = torch.split(
            features_pc,
            nts,
            dim=0,
        )
        for wav_feature_pc, wav_name_no_ext in zip(wav_features_pc, wav_noext_names):
            out_mat_path = os.path.join(variant_dir, f"{wav_name_no_ext}.mat")
            random_filename = str(uuid.uuid4()) + ".mat"
            random_filepath = os.path.join(variant_dir, random_filename)
            if not os.path.exists(variant_dir):
                os.makedirs(variant_dir)
            wav_feature_pc = np.asarray(wav_feature_pc)
            # save data as mat
            hdf5storage.savemat(random_filepath, {"features": wav_feature_pc})
            os.rename(random_filepath, out_mat_path)
    write_summary(
        variant_dir,
        time_window=f"{abs(time_window[0])} second before to {abs(time_window[1])} second after",
        dimensions="[time, pc]",
        sampling_rate=sampling_rate,
        extra=f"""Weights are saved to {out_weights_path}
You can reconstruct (nearly) the original features by reading in the US from the stimname files and multiply by V^T
In python:
feature = hdf5storage.loadmat(stim_file)["features"]
Vt = hdf5storage.loadmat(pca_weights_file)["V"]
original_feature = np.matmul(feature, V.T)
""",
    )
    return variant_dir


def generate_pca_pipeline_from_weights(
    weights_from,
    pc=None,
):
    # pc should be a number
    print(f"Loading weights from {weights_from}")
    weights_mat = hdf5storage.loadmat(weights_from)
    V = weights_mat["V"]
    V_all = weights_mat["V_all"]
    mean = weights_mat["mean"]

    pc = V.shape[1] if pc is None else pc
    pca_pipeline = PCA(n_components=pc, whiten=False)

    mean = torch.as_tensor(mean, dtype=torch.float32)
    V_all = torch.as_tensor(V_all, dtype=torch.float32)
    pca_pipeline.load(mean, V_all)

    return pca_pipeline


if __name__ == "__main__":
    # test
    import numpy as np

    # generate random data
    n_samples = 100
    n_features = 10
    X = np.random.rand(n_samples, n_features)

    # fit PCA
    pc = 3
    pca = PCA(n_components=pc)
    pca.fit(X)
    X_transformed = pca.transform(X)

    hdf5storage.savemat("pca_weights.mat", {"V": V, "V_all": V_all, "mean": mean})
    pass
