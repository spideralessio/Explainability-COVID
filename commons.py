from rdkit import Chem
import numpy as np 
import torch
from torch_geometric.data import Data, DataLoader, Batch

atoms = [1,3,5,6,7,8,9,11,13,14,15,16,17,19,20,25,26,27,30,33,34,35,38,42,47,50,53, 78, 79, 80,83]
CHIRAL_TYPES = len(Chem.rdchem.ChiralType.values)
HYBRIDIZATION_TYPES = len(Chem.rdchem.HybridizationType.values)
SEEN_ATOMS = {k:i for i, k in enumerate(sorted(atoms))}


def atm_data(atm):
    atomic_num = atm.GetAtomicNum()
    atomic_num = np.eye(len(SEEN_ATOMS))[SEEN_ATOMS[atomic_num]]
    chiral = np.eye(CHIRAL_TYPES)[int(atm.GetChiralTag())]
    hybrid = np.eye(HYBRIDIZATION_TYPES)[int(atm.GetHybridization())]
    charge = atm.GetFormalCharge()
    imp_val = atm.GetImplicitValence()
    aromatic = int(atm.GetIsAromatic())
    isotope = atm.GetIsotope()
    mass = atm.GetMass()
    rad_ele = atm.GetNumRadicalElectrons()
    ring = atm.IsInRing()
    data = np.concatenate([atomic_num, [charge,imp_val, aromatic, isotope, mass, rad_ele, ring], chiral, hybrid])
    return data

def gen_graph(mol, act=None):
    #print(act)
    n = mol.GetNumAtoms()
    atoms = []
    #adj = np.zeros((n,n))
    edge_index = []
    for atm in mol.GetAtoms():
        atoms.append(atm_data(atm))
        idx = atm.GetIdx()
        for neig in atm.GetNeighbors():
            neig_idx = neig.GetIdx()
            #adj[idx, neig_idx] = 1
            #adj[neig_idx, idx] = 1
            edge_index.append([idx, neig_idx])
    edge_index = torch.tensor(np.transpose(edge_index)).long()
    #print(edge_index)
    x = torch.tensor(np.array(atoms)).float()
    if act is not None:
        act = torch.tensor([act]).float()
    data = Data(x=x, edge_index=edge_index, y=act)
    return data


#######################################
#             Utilities               #
#######################################
def try_catch(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(e, *args, **kwargs)
        return None


