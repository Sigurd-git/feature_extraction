from scipy.linalg import svd
import hdf5storage
import numpy as np
import os


class PCA:
    def __init__(self, n_components, demean=True, standardize=False):
        self.n_components = n_components
        self.demean = demean
        self.standardize = standardize
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

        if self.demean:
            self.mean = np.mean(X, axis=0)
        else:
            self.mean = 0
        X = X - self.mean

        if self.standardize:
            self.std = np.std(X, axis=0, ddof=1)
        else:
            self.std = 1
        X = X / self.std

        self.U, self.S, self.Vt = svd(X, full_matrices=False)

        self.components = self.Vt[: self.n_components]

    def load(self, mean, std, V):
        """
        Load the mean, standard deviation, and principal components from saved files to prepare for PCA.

        Args:
            mean (numpy.ndarray): The mean of the data.
            std (numpy.ndarray): The standard deviation of the data.
            V (numpy.ndarray): The principal components of the data.

        Returns:
            None
        """
        self.mean = mean
        self.std = std
        self.components = V.T

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
    demean=True,
    std=False,
    variant=None,
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

    Vt = pca_pipeline.Vt
    # asser output_roots are all the same
    assert np.all([output_root == output_roots[0] for output_root in output_roots])
    feature_dir = os.path.join(output_roots[0], "features", feature_name)
    if variant is None:
        variant_dir = os.path.join(feature_dir, f"pc{pc}{appendix}")
    else:
        variant_dir = os.path.join(feature_dir, f"{variant}_pc{pc}{appendix}")

    os.makedirs(variant_dir, exist_ok=True)
    out_mat_path = os.path.join(variant_dir, "pca_weights.mat")
    hdf5storage.savemat(
        out_mat_path,
        {"V": Vt[:pc].T, "mean": pca_pipeline.mean, "std": pca_pipeline.std},
    )
    return pca_pipeline


def apply_pca_pipeline(
    wav_features,
    pc,
    output_roots,
    feature_name,
    wav_noext_names,
    pca_pipeline,
    variant=None,
    appendix="",
):
    if isinstance(output_roots, str):
        # broadcast to list
        output_roots = [output_roots] * len(wav_features)

    feature = np.concatenate(wav_features, axis=0)
    features_pc = pca_pipeline.transform(feature)
    # split back to each wav
    wav_features_pc = np.split(
        features_pc,
        np.cumsum([len(wav_feature) for wav_feature in wav_features])[:-1],
        axis=0,
    )
    for wav_feature_pc, wav_name_no_ext, output_root in zip(
        wav_features_pc, wav_noext_names, output_roots
    ):
        feature_dir = os.path.join(output_root, "features", feature_name)

        if variant is None:
            variant_dir = os.path.join(feature_dir, f"pc{pc}{appendix}")
        else:
            variant_dir = os.path.join(feature_dir, f"{variant}_pc{pc}{appendix}")

        os.makedirs(variant_dir, exist_ok=True)
        out_mat_path = os.path.join(variant_dir, f"{wav_name_no_ext}.mat")
        out_meta_stimuli_path = os.path.join(variant_dir, f"{wav_name_no_ext}.txt")

        if not os.path.exists(variant_dir):
            os.makedirs(variant_dir)

        # save data as mat
        hdf5storage.savemat(out_mat_path, {"features": wav_feature_pc})

        # generate meta file for cochleagram
        with open(out_meta_stimuli_path, "w") as f:
            f.write(f"The shape of this feature is {wav_feature_pc.shape}.")


def generate_pca_pipeline_from_weights(
    weights_from,
    pc=None,
):
    # pc should be a number
    weights_mat = hdf5storage.loadmat(weights_from)
    V = weights_mat["V"]
    mean = weights_mat["mean"]
    std = weights_mat["std"]
    if mean == 0:
        demean = False
    else:
        demean = True
    if std == 1:
        std = False
    else:
        std = True
    pc = V.shape[1] if pc is None else pc
    pca_pipeline = PCA(n_components=pc, demean=demean, standardize=std)
    pca_pipeline.load(mean, std, V)

    return pca_pipeline
