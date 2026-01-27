import dynamo.dynamo as dym
import numpy as np
import dynamo_rp.rp_utility as rpt
from dynamo_rp import parameters as pm

# TODO Only functions that deals with dym objects should be here.  


#TODO move this function to rp_utility
def get_bounding_rp_from_chain(chain, model_params):
    """
    Gets the appropriate rp(triplet) for the first and last module in the chain.
    Extends the chain from [A, ... Z] to [A, A, ... Z, Z] to get the rp(triplet)
    for the first and last module. In the case of a junction module, [U_junc_V, ..., X_junc_Y], 
    we use the following extension [U, U_junc_V, ..., X_junc_Y, Y] to get the rp(triplet)
    for the first module.
    Parameters:
      chain (list): A list of strings representing the modules in the chain.
                    ["D14", "D14_j1_D14", ...]                 
      model_params (dict): A dictionary containing the parameters.
    Returns:
      l_bound (str): The rp(triplet) for the first module in the chain.
      r_bound (str): The rp(triplet) for the last module in the chain.
    """
    l_mod, c_mod, r_mod = chain[0].split("_")[0], chain[0], chain[1]
    l_base = pm.rp_modules_df.query(
        f"L == '{l_mod}' and C == '{c_mod}' and R == '{r_mod}'"
    )
    l_mod, c_mod, r_mod = chain[-2], chain[-1], chain[-1].split("_")[-1]
    r_base = pm.rp_modules_df.query(f"L == '{l_mod}' and C == '{c_mod}'")
    r_base = pm.rp_modules_df.query(
        f"L == '{l_mod}' and C == '{c_mod}' and R == '{r_mod}'"
    )
    l_bound, r_bound = list(dict(l_base.T).keys())[0], list(dict(r_base.T).keys())[0]
    return l_bound, r_bound


# TODO Move this function to rp_utility 
def get_rps_from_chain(chain, model_params):
    """
    Gets the rp(triplet) for each module in the chain. Padding the ends of the chain with
    homologous modules.
    Parameters:
      chain (list): A list of strings representing the modules in the chain.
      model_params (dict): A dictionary containing the parameters.
    """
    l_bound, r_bound = get_bounding_rp_from_chain(chain, model_params)
    bulk = [l_bound]
    for i in range(1, len(chain) - 1):
        bulk.append(pm.triple_to_rp["-".join(chain[i - 1 : i + 2])])
    bulk.append(r_bound)
    return bulk


# TODO Move this function to rp_utility
def get_model_params(folder):
    """
    Parses json files to create a dictionary containing the parameters.
    Expects the each json file to be named after the protein it contains.
    Parameters:
      folder (str): The folder containing the json files.
    Returns:
      model_params (dict): A dictionary containing the parameters.
                          the keys are the protein names.
    """
    import json
    import glob

    file_names = glob.glob(folder + "/*.json")
    model_params = dict()
    for file_name in file_names:
        try:
            f = open(file_name)
            protein_name = file_name.split("/")[-1].split(".")[0]
            model_params[protein_name] = json.loads(f.read())
            f.close()
        except:
            print(f"Error reading {file_name}")
            return None
    return model_params


def get_prob_pos_from_kwargs(mu=None, cov=None, w=None):
    """
    Creates a ProbPos object from the kwargs.
    Parameters (kwargs):
      mu (np.array): The mean of the distribution. default: np.zeros((1, 3))
      cov (np.array): The covariance of the distribution. default: np.zeros((1, 3, 3))
      w (np.array): The weight of the distribution. default: np.array([1.0])
    Returns:
      prob_pos (ProbPos): A ProbPos object.
    """
    if mu is None:
        mu = np.zeros((1, 3))
    if mu is None:
        cov = np.zeros((1, 3, 3))
    if w is None:
        w = np.array([1.0])
    return dym.ProbPos(np.array(mu), np.array(cov), np.array(w))


def get_general_module_from_rp(rp, model_params):
    """
    Creates a GeneralModule from the rp(triplet).
    Parameters:
      rp (str): The rp (triplet) of the module.
      model_params (dict): A dictionary containing the parameters.
    Returns:
      module (GeneralModule): A GeneralModule object.
    """
    module_params = model_params[rp]
    ref_points = []
    p_vecs = []
    ref_frames = []
    for params in module_params["ref_points_params"]:
        ref_points.append(get_prob_pos_from_kwargs(**params))

    for p_vec_params, ref_frame in zip(
        module_params["p_vec_params"], module_params["next_ref_frames"]
    ):
        p_vecs.append(get_prob_pos_from_kwargs(**p_vec_params))
        ref_frame = np.array(ref_frame)
        ref_frames.append(ref_frame)

    module = dym.GeneralModule(p_vecs, ref_frames)
    module.tracked_points = ref_points
    return module


def get_general_modules_from_chain(chain, model_params):
    """
    Creates a list of GeneralModules from the chain.
    Parameters:
      chain (list): A list of strings representing the modules in the chain.
                    e.g. ["D14", "D14_j1_D14", ...]
      model_params (dict): A dictionary containing the parameters.
    Returns:
      modules (list): A list of GeneralModule objects.
    """
    rps = get_rps_from_chain(chain, model_params)
    modules = [get_general_module_from_rp(rp, model_params) for rp in rps]
    return modules


def get_gaussian_mixture_from_prob_pos(prob_pos, reduce=False):
    """
    Gets a GaussianMixture object from a ProbPos object.
    Parameters:
        prob_pos (ProbPos): A ProbPos object.
        reduce (bool): Whether to reduce the ProbPos object to a single Gaussian.
    Returns:
        mixture (GaussianMixture): A GaussianMixture object.
    """
    if reduce:
        covs = np.array([prob_pos.cov()])
        mus = np.array([prob_pos.mu()])
        weights = np.array([1.0])

    else:
        mus = prob_pos.mus
        covs = prob_pos.covs
        weights = prob_pos.weights
    return rpt.get_mixture_from_params(mus, covs, weights)
