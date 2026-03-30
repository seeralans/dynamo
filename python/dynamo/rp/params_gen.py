import numpy as np
import json
from sklearn.mixture import GaussianMixture

class NumpyEncoder(json.JSONEncoder):
    """
    A class to encode numpy arrays into json.
    Taken from KarlB's andwer on
    https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def get_gm_params(data, num_components, **kwargs):
    """
    Get the parameters of a Gaussian Mixture model.
    Parameters:
      data (np.array): The data to be used for the model.
      num_components (int): The number of components to be used.
      **kwargs: The keyword arguments for the GaussianMixture object.
    Returns:
      out (dict): A dictionary containing the parameters.
      out = {"mu": mus, "cov": covs, "w": weights}
    """
    gm = GaussianMixture(num_components, **kwargs)
    gm.fit(data)
    return {"mu": gm.means_, "cov": gm.covariances_, "w": gm.weights_}



def get_k_fold_validation(data, n_components, **kwargs):
    """
    Get the k-fold validation score for the data.
    Parameters:
      data (np.array): The data to be used for the validation.
      n_components (int): The number of components to be used.
      **kwargs: The keyword arguments for the KFold object.
    Returns:
      out (np.array): The validation scores.
    """
    from sklearn.model_selection import KFold
    kf = KFold(**kwargs)
    out = []
    for i, (train_index, test_index) in enumerate(kf.split(data)):
        gm = GaussianMixture(n_components=n_components)
        gm.fit(data[train_index])
        out.append(gm.score(data[test_index]))
    return np.array(out)

def get_best_n_components(data, num_eval=20, n_splits=5, **kwargs):
    """
      Get the best number of components for the data, using k-fold validation.
      Parameters:
        data (np.array): The data to be used for the validation.
      Optional:
        num_eval (int): The number of evaluations to be done.
        n_splits (int): The number of splits to be used for the k-fold validation.
        **kwargs: The keyword arguments for the KFold object.
      Returns:
        out (int): The best number of components.
    """
    k_dat = []
    for k in range(1, n_splits+1):
        terms = []
        for _ in range(num_eval):
            val = get_k_fold_validation(data, k, n_splits=n_splits, shuffle=True).mean()
            terms.append(val)
        k_dat.append(np.array(terms).mean())
    # k5_dat = [np.array([get_k_fold_validation(data, k, n_splits=5, shuffle=True).mean()
    #                     for _ in range(num_eval)]).mean() for k in range(1, 6)]
    k_max = k_dat.index(max(k_dat)) + 1
    return k_max

# TODO point to how things are aligned
def compute_internal_ref_frame(hels, stack=False):
    """
    Get the internal reference frame for a set of helices. 
    Parameters:
      hels (np.array): The helices to be used.
    Optional:
      stack (bool): Whether to stack the vectors into a single array.
    Returns:
      out (np.array): The internal reference frame.
    """

    k_vecs = hels.mean(axis=(1))[:, 0] - hels.mean(axis=(1))[:, 1]
    k_vecs /= np.linalg.norm(k_vecs, axis=1)[:, None]
    jt_vecs = hels[:, 0:2, :].mean(axis=(1, 2)) - hels[:, 2:, :].mean(axis=(1, 2))
    i_vecs = np.cross(k_vecs, jt_vecs)
    i_vecs /= np.linalg.norm(i_vecs, axis=1)[:, None]
    j_vecs = np.cross(k_vecs, i_vecs)
    if stack:
        return np.stack((i_vecs, j_vecs, k_vecs), axis=2)
    return i_vecs, j_vecs, k_vecs


# TODO clean up
def align_internal(centre_hels, other_hels,):
    """
    Align a set of helices to a reference frame. Centre hels is
    the reference frame to be standardised. 
    Parameters:
      centre_hels (np.array): The helices to be used as the reference frame.
      other_hels (np.array): The helices to be aligned.
    """
    from Bio.SVDSuperimposer import SVDSuperimposer
    centroids = centre_hels.mean(axis=(1, 2))
    i_vecs, j_vecs, k_vecs = compute_internal_ref_frame(centre_hels)
    align_points = np.transpose(
      np.stack((centroids,
                centroids + i_vecs,
                centroids + j_vecs,
                centroids + k_vecs,
                centroids + i_vecs + k_vecs,
                centroids + i_vecs + j_vecs,
                centroids + k_vecs + j_vecs,
                centroids + i_vecs + j_vecs + k_vecs),
               axis=2), (0, 2, 1))

    # Unit cube
    reference = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 1.0],
        [1.0, 1.0, 1.0],
    ])
    all_rot_trans = []
    sup = SVDSuperimposer()
    for i, points in enumerate(align_points):
        sup.set(reference, points)
        sup.run()
        all_rot_trans.append(sup.get_rotran())

    rot_trans = []
    hels_trans = []
    other_hels_trans = []
    centre_hels_trans = None
    for m, hels in enumerate([centre_hels] + other_hels):
        hels_trans = [] 
        for i, rot_trans in enumerate(all_rot_trans):
          hels_upper = np.dot(hels[i, :, 0], rot_trans[0]) + rot_trans[1]
          hels_lower = np.dot(hels[i, :, 1], rot_trans[0]) + rot_trans[1]
          hels_trans.append((hels_upper, hels_lower))
        hels_trans = np.array(hels_trans)
        hels_trans = np.transpose(hels_trans, (0, 2, 1, 3))
        if m == 0:
            centre_hels_trans = hels_trans
        else:
            other_hels_trans.append(hels_trans)
    return centre_hels_trans, other_hels_trans

