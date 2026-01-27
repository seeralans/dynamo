import json
import Bio.PDB
import numpy as np
from string import Template
from dynamo_rp import parameters as pm

CART_RELAX_STRING = """
<ROSETTASCRIPTS>
  <MOVERS>
    <FastRelax name="cartrelax" cartesian="true">
      <MoveMap>
        $movemap
      </MoveMap>
    </FastRelax>
  </MOVERS>
  <PROTOCOLS>
    <Add mover="cartrelax"/>
  </PROTOCOLS>
</ROSETTASCRIPTS>
"""
SPAN_STRING = Template('<Span begin="$start" end="$end" chi="$chi" bb="$bb"/>')
CART_RELAX_TEMPLATE = Template(CART_RELAX_STRING)


def construct_cart_relax(chain, buff=3, out_file=None, cap=False):
    """
    Construct a cartesian relax xml script for a given chain of modules.
    Parameters:
        chain (list): A list containing the modules e.g. ["D49", "D49", "D49"]
    Optional:
        buff (int): The number of amino acids above and below the connection,
                    e.g. buff=3 (default) will relax 6
        out_file (str): The filename where the xml file shall be saved.
    Returns:
        script (str): The xml script as a string.
    """
    from dynamo_rp import parameters as pm

    residue_count = np.array([pm.module_lengths[mod] for mod in chain]).cumsum()
    span_strings = [SPAN_STRING.substitute(start=1, end=residue_count[-1], chi=0, bb=0)]
    for i in range(len(chain) - 1):
        span_strings.append(
            SPAN_STRING.substitute(
                start=residue_count[i] - buff, end=residue_count[i] + buff, chi=1, bb=1
            )
        )
    movemap = "\n".join(span_strings)
    script = CART_RELAX_TEMPLATE.substitute(movemap=movemap)
    if out_file is not None:
        with open(out_file, "w") as f:
            f.write(script)
    return script


def read_pdb(pdb_id, input_file, permissive=0):
    """
    Read a pdb file and return a structure object.
    Parameters:
        pdb_id (str): The pdb id of the protein.
        input_file (str): The path to the pdb file.
    Optional:
        permissive (int): Whether to be permissive when reading the pdb file.
    Returns:
        structure (Bio.PDB.Structure.Structure): The structure object.
    """
    parser = Bio.PDB.PDBParser(permissive)
    structure = parser.get_structure(pdb_id, input_file)
    return structure


def get_residue_count(pdb):
    """
    Get the number of residues in a pdb file.
    Parameters:
        pdb (Bio.PDB.Structure.Structure): The structure object.
    """
    return sum([len(c.child_list) for c in pdb.child_list[0].child_list])


def mod_to_pdb_single_file(dh, elfin_aligned_pdb_folder):
    """
    Get the path to the pdb file for a single module.
    Parameters:
        dh (str): The dh of the module.
        elfin_aligned_pdb_folder (str): The path to the folder
        containing the pdb files. Must be in the format:
        elfin_aligned_pdb_folder
          |- singles
            |- D49.pdb
          |- doubles
            |- D49-D49.pdb
    Returns:
        pdb_file (str): The path to the pdb file.
    """
    return elfin_aligned_pdb_folder + r"//singles/" + f"{dh}.pdb"


def mod_to_pdb_double_file(dh1, dh2, elfin_aligned_pdb_folder):
    """
    Get the path to the pdb file for a pair of module.
    Parameters:
        dh1 (str): The dh of the first module.
        dh2 (str): The dh of the second module.
        elfin_aligned_pdb_folder (str): The path to the folder
        containing the pdb files. Must be in the format:
        elfin_aligned_pdb_folder
          |- singles
            |- D49.pdb
          |- doubles
            |- D49-D49.pdb
    Returns:
        pdb_file (str): The path to the pdb file.
    """
    return elfin_aligned_pdb_folder + r"/doubles/" + f"{dh1}-{dh2}.pdb"


def construct_large_protein(chain, out_file, elfin_aligned_pdb_folder, cap=False):
    """
    Construct a large protein from a chain of modules.
    Parameters:
        chain (list): A list containing the modules e.g. ["D49", "D49", "D49"]
        out_file (str): The filename of the pdb
        elfin_aligned_pdb_folder (str): The path to the folder
        containing the pdb files. Must be in the format:
        elfin_aligned_pdb_folder
          |- singles
            |- D49.pdb
          |- doubles
            |- D49-D49.pdb
        These pdbs of singles and doubles must be aligned appropriately. For example,
        D49.pdb must be aligned to the first D49 of D49-D49.pdb.
    Returns:
        None
    """
    out = Bio.PDB.Chain.Chain("A")
    residue_number = 1

    a_name = chain[0]
    A = (
        read_pdb(a_name, mod_to_pdb_single_file(a_name, elfin_aligned_pdb_folder))
        .child_list[0]
        .child_list[0]
    )
    detached_res = [A.child_list[i] for i in range(len(A.child_list))]

    def append_residues(module, residue_number):
        detached_res = [module.child_list[i] for i in range(len(module.child_list))]
        for res in detached_res:
            module.detach_child(res.id)
        for res in detached_res:
            res.id = (res.id[0], residue_number, res.id[2])
            out.add(res)
            residue_number += 1
        return residue_number

    residue_number = append_residues(A, residue_number)
    for i in range(len(chain) - 1):
        a_name = chain[i]
        b_name = chain[i + 1]
        A = (
            read_pdb(a_name, mod_to_pdb_single_file(a_name, elfin_aligned_pdb_folder))
            .child_list[0]
            .child_list[0]
        )
        B = (
            read_pdb(b_name, mod_to_pdb_single_file(b_name, elfin_aligned_pdb_folder))
            .child_list[0]
            .child_list[0]
        )
        AB = (
            read_pdb(
                a_name + b_name,
                mod_to_pdb_double_file(a_name, b_name, elfin_aligned_pdb_folder),
            )
            .child_list[0]
            .child_list[0]
        )

        a_len = len(A.child_list)
        b_len = len(B.child_list)

        # current positions of ca in A module
        c_a_cas = [a for r in out.child_list[-a_len:] for a in r if a.name == "CA"]

        # default positions of ca in A module
        a_cas = [a for r in A.child_list for a in r if a.name == "CA"]

        # default positions of ca in B module
        b_cas = [a for r in B.child_list for a in r if a.name == "CA"]

        ab_cas_a = [
            a for r in AB.child_list[: len(A.child_list)] for a in r if a.name == "CA"
        ]
        ab_cas_b = [
            a for r in AB.child_list[len(A.child_list) :] for a in r if a.name == "CA"
        ]

        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(c_a_cas, ab_cas_a)
        rot_tran_ab_to_a = super_imposer.rotran

        super_imposer.set_atoms(ab_cas_b, b_cas)
        rot_tran_b_to_ab = super_imposer.rotran

        B.transform(*rot_tran_b_to_ab)
        B.transform(*rot_tran_ab_to_a)
        residue_number = append_residues(B, residue_number)

    if cap:
        n_cap_name = chain[0].split("_")[0]
        c_cap_name = chain[-1].split("_")[-1]

        n_cap_file = elfin_aligned_pdb_folder + r"/caps/" + f"{n_cap_name}_NI.pdb"
        c_cap_file = elfin_aligned_pdb_folder + r"/caps/" + f"{c_cap_name}_IC.pdb"

        NC = read_pdb(n_cap_name, n_cap_file).child_list[0].child_list[0]
        CC = read_pdb(c_cap_name, c_cap_file).child_list[0].child_list[0]

        n_len = len(NC.child_list) // 2
        c_len = len(CC.child_list) // 2

        # current position of last two helicies
        c_c_cas = [a for r in out.child_list[-c_len:] for a in r if a.name == "CA"]

        # current  cas in C cap
        c_cas = [a for r in CC.child_list[:c_len] for a in r if a.name == "CA"]

        super_imposer = Bio.PDB.Superimposer()
        super_imposer.set_atoms(c_c_cas, c_cas)
        rot_tran_c_to_c = super_imposer.rotran
        CC.transform(*rot_tran_c_to_c)
        detached_res = CC.child_list[-c_len:]
        for res in detached_res:
            CC.detach_child(res.id)
        for res in detached_res:
            res.id = (res.id[0], residue_number, res.id[2])
            out.add(res)
            residue_number += 1

        n_c_cas = [a for r in out.child_list[:n_len] for a in r if a.name == "CA"]

        n_cas = [a for r in NC.child_list[n_len:] for a in r if a.name == "CA"]

        super_imposer.set_atoms(n_c_cas, n_cas)
        rot_tran_n_to_n = super_imposer.rotran
        NC.transform(*rot_tran_n_to_n)
        detached_res = NC.child_list[:n_len]
        for res in detached_res:
            NC.detach_child(res.id)
        for i, res in enumerate(detached_res):
            res.id = (res.id[0], residue_number, res.id[2])
            out.insert(i, res)
            residue_number += 1



    detached_res = out.child_list[:-1]
    for res in detached_res:
        out.detach_child(res.id)

    oout = Bio.PDB.Chain.Chain("A")
    res_number = 1
    for res in detached_res:
        res.id = (res.id[0], res_number, res.id[2])
        oout.add(res)
        res_number += 1
        

        

    io = Bio.PDB.PDBIO()
    io.set_structure(oout)
    print("Saving to ", out_file)
    io.save(out_file)
