import numpy as np
from dynamo_rp import parameters as pm
from dynamo_rp import hull_generation as hg
from sklearn.mixture import GaussianMixture



def get_res_counts_for_chain(chain):
    """
    Computes the cumulative sum of the number of residues in each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of cumulative sum of the number of residues in each module in the chain.
    """
    return list(np.array([pm.module_lengths[mod] for mod in chain]).cumsum())


def get_helical_counts_for_chain(chain):
    """
    Gets the cumulative sum of the number of helices in each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of cumulative sum of the number of helices in each module in the chain.
    """
    return list((np.array([pm.module_num_helices[mod] for mod in chain]).cumsum()))


def get_helical_residue_location_for_chain(chain):
    """
    Gets the helical residue location for each module in the chain.
    Parameters:
      chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
    Returns:
      a list of lists of helical residue locations for each module in the chain.
    """
    res_count = [0] + get_res_counts_for_chain(chain)
    helices = [
        [
            [idx + res_count[i] for idx in pair]
            for pair in pm.module_helical_start_ends[mod]
        ]
        for i, mod in enumerate(chain)
    ]
    return helices


def get_rp_helices(rp, conf, folder, start_end=(None, None)):
    """
    Get the helices for a given rp and conf in a given folder.
    Folder structure must be: 
    folder
    ├── rp1
    │   ├── rp1_0001.vtx
    The format of the vtx file can be seen in
    hull_generation.write_list_of_list_vtx
    Parameters:
      rp (str): rp name
      conf (int): conf number
      folder (str): folder name
    Optional:
      start_end (tuple): start and end helix index
    Returns:
      a numpy array of shape (num_helices, 2, 3)
    """
    file_name = f"{folder}/{rp}/{rp}_{conf:04}.vtx"
    point_list = hg.read_list_of_list_vtx(file_name)
    num_helices = len(point_list) // 2
    out = []
    for i in range(num_helices):
        u = np.array(point_list[i][1:])
        l = np.array(point_list[i + num_helices][1:])
        out.append(np.array([u, l]))

    coords = np.array(out)

    return coords[start_end[0] : start_end[1], :, :]


def get_rp_helices_all(rp, conf_range, folder, array_out=True, **kwargs):
    """
    Get the helices for a given rp and confrange in a given folder.
    Folder structure must be: 
    folder
    ├── rp1
    │   ├── rp1_0001.vtx
    Parameters:
      rp (str): rp name
      conf_range (range): range of confs
      folder (str): folder name
    Optional:
      array_out (bool): return as numpy array
      **kwargs: passed to get_rp_helices
    Returns:
      a numpy array of shape (num_confs, num_helices, 2, 3)
    """
    out = []
    for conf in conf_range:
        coords = get_rp_helices(rp, conf, folder, **kwargs)
        out.append(coords)
    if array_out:
        return np.array(out)
    else:
        return out


def split_hels_into_modules(rp, hels):
    """
    Split the helices into modules based on the rp.
    Assumes the helices are in the order of the rp,
    and that caps are included. Uses the rp_module_list database.
    Parameters:
        rp (str): rp name
        hels (np.array): helices of shape (num_helices, 2, 3)
    Returns:
        a tuple of helices for each module in the rp.
        (hels_a_cap, hels_a, hels_b, hels_c, hels_c_cap)
    """
    hel_idxs = np.cumsum([pm.module_num_helices[mod] for mod in pm.rp_module_list[rp]])
    hels_a_cap = hels[:, 0 : hel_idxs[1]]
    hels_a = hels[:, hel_idxs[0] : hel_idxs[1]]
    hels_b = hels[:, hel_idxs[1] : hel_idxs[2]]
    hels_c = hels[:, hel_idxs[2] : hel_idxs[3]]
    hels_c_cap = hels[:, hel_idxs[2] :]
    return hels_a_cap, hels_a, hels_b, hels_c, hels_c_cap


def load_prot_names(mod, length, folder):
    """
    Load the protein names for a given module and length.
    Folder structure must be:
    folder
    ├── modxlength.txt
    Parameters:
        mod (str): starting module name
        length (int): length of the rp
        folder (str): folder name
    Returns:
        a dictionary of protein names for each module in the rp.
    """
    file_name = f"{mod}x{length}.txt"
    lines = [l.strip().split() for l in open(folder + "/" + file_name, "r").readlines()]
    module_list = {l[0]: l[1].split("-") for l in lines}
    return module_list


def get_mixture_from_params(mus, covs, weights):
    """
    Get a GaussianMixture object from the parameters.
    Parameters:
        mus (np.array): means of shape (num_components, num_features)
        covs (np.array): covariances of shape (num_components, num_features, num_features)
        weights (np.array): weights of shape (num_components)
    Returns:
        a GaussianMixture object
    """
    gm = GaussianMixture(n_components=len(mus))
    gm.means_init = mus
    gm.fit(np.zeros((max(len(mus), 2), 3)))
    gm.weights_ = weights
    gm.means_ = mus
    gm.covariances_  = covs
    gm.converged_ = True
    return gm


def spherical_to_cart(r, theta, phi):
    """
    Convert from spherical to Cartesian coordinates.
    Parameters:
        r (float): radius
        theta (float): polar angle
        phi (float): azimuthal angle
    Returns:
        a tuple of Cartesian coordinates (x, y, z)
    """
    from numpy import cos, sin
    z = r * cos(theta)
    x = r * sin(theta) * cos(phi)
    y = r * sin(theta) * sin(phi)
    return x, y, z


def cart_to_spherical(x, y, z):
    """
    Convert from Cartesian to spherical coordinates.
    Parameters:
        x (float): x coordinate
        y (float): y coordinate
        z (float): z coordinate
    Returns:
        a tuple of spherical coordinates (r, theta, phi)
    """
    from numpy import cos, sin, arccos, arctan2
    r = np.sqrt(z**2 + x**2 + y**2)
    theta = arccos(z / r)
    phi = arctan2(y, x)
    return r, theta, phi
