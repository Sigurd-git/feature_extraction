from scipy.linalg import svd
import hdf5storage
import numpy as np
import os
from utils.shared import write_summary
from sklearn.decomposition import PCA as PCA_sklearn


class PCA:
    def __init__(self, n_components, whiten=False):
        self.n_components = n_components
        self.whiten = whiten
        self.mean = None
        self.std = None
        self.components = None
        # 新增保存U, S, Vt的属性
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, X):
        """
        Learn the PCA weights of dataset X and save U, S, Vt.

        Parameters:
        - X: numpy array of shape (n_samples, n_features)
        """

        if self.whiten:
            self.std = np.std(X, axis=0)

        n_features = X.shape[1]
        pca_sklearn = PCA_sklearn(n_components=n_features, whiten=self.whiten)
        pca_sklearn.fit(X)
        self.Vt = pca_sklearn.components_
        self.components = self.Vt[: self.n_components]
        self.mean = pca_sklearn.mean_

    def load(self, mean, std, V_all):
        """
        Load the mean, standard deviation, and principal components from saved files to prepare for PCA.

        Args:
            mean (numpy.ndarray): The mean of the data.
            std (numpy.ndarray): The standard deviation of the data.
            V_all (numpy.ndarray): The principal components of the data.

        Returns:
            None
        """
        self.mean = mean
        self.std = std
        self.Vt = V_all.T
        self.components = V_all[:, : self.n_components].T

    def transform(self, X):
        """
        Apply PCA weights to the new data set X and perform SVD decomposition.

        Parameters:

        - X: numpy array of shape (n_samples, n_features)

        Return value:

        - X_transformed: numpy array of shape (n_samples, n_components)

        - U_new: U matrix obtained by SVD decomposition

        - S_new: The singular value obtained by SVD decomposition

        - Vt_new: Vt matrix obtained by SVD decomposition
        """
        # Go to the average

        X = X - self.mean
        # Standardization

        X = X / self.std
        # Convert the PCA weight obtained during training
        X_transformed = np.matmul(X, self.components.T)

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
    output_roots,
    feature_name,
    variant,
    demean=True,
    std=False,
    appendix="",
):
    # pc should be a number
    # concatenate all features in time
    if isinstance(output_roots, str):
        # broadcast to list
        output_roots = [output_roots] * len(wav_features)

    feature = np.concatenate(wav_features, axis=0)

    pca_pipeline = PCA(n_components=pc, demean=demean, standardize=std)
    pca_pipeline.fit(feature)

    Vt = pca_pipeline.components
    Vt_all = pca_pipeline.Vt
    # asser output_roots are all the same
    assert np.all([output_root == output_roots[0] for output_root in output_roots])
    feature_dir = os.path.join(output_roots[0], "features", feature_name)
    variant_dir = os.path.join(feature_dir, f"{variant}{appendix}")

    os.makedirs(variant_dir, exist_ok=True)
    out_mat_path = os.path.join(variant_dir, "metadata", "pca_weights.mat")
    hdf5storage.savemat(
        out_mat_path,
        {
            "V": Vt.T,
            "V_all": Vt_all.T,
            "mean": pca_pipeline.mean,
            "std": pca_pipeline.std,
        },
    )
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
    feature_dir = os.path.join(output_root, "features", feature_name)

    if (variant is None) or (variant == "") or (variant == "original"):
        variant_dir = os.path.join(feature_dir, f"pc{pc}{appendix}")
    else:
        variant_dir = os.path.join(feature_dir, f"{variant}_pc{pc}{appendix}")

    os.makedirs(os.path.join(variant_dir, "metadata"), exist_ok=True)
    out_weights_path = os.path.join(variant_dir, "metadata", "pca_weights.mat")
    Vt = pca_pipeline.components
    Vt_all = pca_pipeline.Vt
    hdf5storage.savemat(
        out_weights_path,
        {
            "V": Vt.T,
            "V_all": Vt_all.T,
            "mean": pca_pipeline.mean,
            "std": pca_pipeline.std,
        },
    )
    if not meta_only:
        feature = np.concatenate(wav_features, axis=0)
        features_pc = pca_pipeline.transform(feature)
        # split back to each wav
        wav_features_pc = np.split(
            features_pc,
            np.cumsum([len(wav_feature) for wav_feature in wav_features])[:-1],
            axis=0,
        )
        for wav_feature_pc, wav_name_no_ext in zip(wav_features_pc, wav_noext_names):
            out_mat_path = os.path.join(variant_dir, f"{wav_name_no_ext}.mat")

            if not os.path.exists(variant_dir):
                os.makedirs(variant_dir)

            # save data as mat
            hdf5storage.savemat(out_mat_path, {"features": wav_feature_pc})
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
    std = weights_mat["std"]
    if np.all(mean == 0):
        demean = False
    else:
        demean = True
    if np.all(std == 1):
        std = False
    else:
        std = True
    pc = V.shape[1] if pc is None else pc
    pca_pipeline = PCA(n_components=pc, demean=demean, standardize=std)
    pca_pipeline.load(mean, std, V_all)

    return pca_pipeline


if __name__ == "__main__":
    # test
    import numpy as np

    # generate random data
    n_samples = 100
    n_features = 10
    demean = True
    std = False
    X = np.random.rand(n_samples, n_features)

    # fit PCA
    pc = 3
    pca = PCA(n_components=pc)
    pca.fit(X)
