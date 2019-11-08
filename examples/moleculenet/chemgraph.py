"""
Utility functions for creating featurizations of small molecules.
"""
from itertools import combinations

import networkx as nx
import numpy as np
from rdkit import Chem


def atom_graph(mol: Chem.rdchem.Mol):
    """
    Generates the atom graph from an RDKit Mol object.

    Function taken from https://github.com/maxhodak/keras-molecules/pull/32/files.
    """
    if mol:
        G = nx.Graph()
        for atom in mol.GetAtoms():
            G.add_node(
                atom.GetIdx(),
                atomic_num=atom.GetAtomicNum(), #this should be instantiated once, and later reused for defining the feature vector
                formal_charge=atom.GetFormalCharge(),
                chiral_tag=atom.GetChiralTag(),
                hybridization=atom.GetHybridization(),
                num_explicit_hs=atom.GetNumExplicitHs(),
                is_aromatic=atom.GetIsAromatic(),
                mass=atom.GetMass(),
                implicit_valence=atom.GetImplicitValence(),
                total_hydrogens=atom.GetTotalNumHs(),
                features=np.array(
                    [
                        atom.GetAtomicNum(),
                        atom.GetFormalCharge(),
                        atom.GetChiralTag(),
                        atom.GetHybridization(),
                        atom.GetNumExplicitHs(),
                        atom.GetIsAromatic(),
                        atom.GetMass(),
                        atom.GetImplicitValence(),
                        atom.GetTotalNumHs(),
                    ]
                ),
            )
        for bond in mol.GetBonds():
            G.add_edge(
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                bond_type=bond.GetBondType(),
            )
        return G


def bond_graph(mol: Chem.rdchem.Mol):
    """
    Generates the bond graph from an RDKit Mol object.

    Here, unlike the atom gaph, bonds are nodes, and are
    connected to each other by atoms.

    :returns: a NetworkX graph.
    """
    if mol:
        G = nx.Graph()
        for bond in mol.GetBonds():
            G.add_node(
                (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                bond_type=bond.GetBondTypeAsDouble(),
                aromatic=bond.GetIsAromatic(),
                stereo=bond.GetStereo(),
                in_ring=bond.IsInRing(),
                is_conjugated=bond.GetIsConjugated(),
                features = [
                    bond.GetBondTypeAsDouble(),
                    int(bond.GetIsAromatic()),
                    # bond.GetStereo(),
                    int(bond.IsInRing()),
                    int(bond.GetIsConjugated()),
                ]
            )

        for atom in mol.GetAtoms():
            bonds = atom.GetBonds()
            if len(bonds) >= 2:
                for b1, b2 in combinations(bonds, 2):
                    n1 = (b1.GetBeginAtomIdx(), b1.GetEndAtomIdx())
                    n2 = (b2.GetBeginAtomIdx(), b2.GetEndAtomIdx())
                    joining_node = list(set(n1).intersection(n2))[0]
                    G.add_edge(n1, n2, atom=joining_node)
                    G.add_edge(n2, n1)
        return G


def atom_transformers(encs):
    """
    Utility function intended to build a transformer dictionary that
    gets passed into the chem.node_feats() function.
    """
    tfm = {
        "atomic_num": None,
        "chiral_tag": encs["chiral_tag"],
        "formal_charge": int,
        "num_explicit_hs": int,
        "is_aromatic": int,
        # "implicit_valence": int,
        # "total_hydrogens": int,
    }
    return tfm


def bond_transformers(encs):
    tfm = {
        # "is_conjugated": int,
        # "in_ring": int,
        # "aromatic": int,
        "stereo": encs["stereo"],
        "bond_type": encs["bond_type"],
    }
    return tfm
