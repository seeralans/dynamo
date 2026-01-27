import numpy as np
from dynamo_rp import rp_utility as rpt


def write_list_of_list_vtx(file_name, a_list):
    """
    Write a list of list to a file. Each list is a line in the file.
    Parameters:
      file_name: file name
      a_list: a list of list
    Return:
      None
    """
    import os

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    with open(file_name, "w") as fp:
        lines = [",".join([str(s) for s in item]) for item in a_list]
        fp.write("\n".join(lines))

def read_list_of_list_vtx(file_name):
    """
    Read a list of list from a file. Each list is a line in the file.
    Parameters:
      file_name: file name
    Return:
      a list of list
    """
    with open (file_name, "r") as fp:
        lines = fp.readlines()
        lines = [[x if i == 0 else float(x) for i, x  in enumerate(l.strip().split(","))] for l in lines]
    return lines

def write_list_of_list_edges(file_name, a_list):
  """
  Write a list of list to a file. Each list is a line in the file.
  Parameters:
    file_name: file name
    a_list: a list of list
  Return:
    None
  """
  os.makedirs(os.path.dirname(file_name), exist_ok=True)
  with open (file_name, "w") as fp:
    lines = [",".join([str(s) for s in item])for item in a_list]
    fp.write("\n".join(lines))

def read_list_of_list_edges(file_name):
  """
  Read a list of list from a file. Each list is a line in the file.
  Parameters:
    file_name: file name
  Return:
    a list of list
  """
  with open (file_name, "r") as fp:
    lines = fp.readlines()
    lines = [l.strip().split(",") for l in lines]
  return lines


def get_hull_points_from_pdb(module_chain, pdb_fname, **kwargs):
    """
    Get the hull points from pdb file
    Parameters:
      module_chain: a list of modules present e.g. ['NcapD4', 'D4', 'D4', 'D4', 'CcapD4']
      pdb_fname: pdb filename
    Optional:
      **kwargs: keyword arguments for get_coords_calphas
    """
    starts_ends = np.concatenate(
        rpt.get_helical_residue_location_for_chain(module_chain)
    )

    # The negative one is due to counting from 1 in pdb file
    coords = get_coords_calphas(starts_ends.ravel()-1, pdb_fname, **kwargs)
    starts = coords[0::2]
    ends = coords[1::2]
    return starts, ends

def get_coords_calphas(idxs, pdb_fname, chain_label="a"):
    """
    Get the coordinates of C-alpha atoms from pdb file.
    Parameters:
        idxs: a list of residue indexes
        pdb_fname: pdb filename
    Optional:
        chain_label: label of chain in pdb file
    """
    import Bio.PDB as bp
    parser = bp.PDBParser()
    structure = parser.get_structure("rp", pdb_fname)
    ar = np.array([res["CA"].coord for res in structure[0][chain_label]])
    return ar[idxs]

