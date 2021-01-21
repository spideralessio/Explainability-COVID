import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.data import Data, DataLoader, Batch
from commons import SEEN_ATOMS, atoms, try_catch, gen_graph

def visualize(smiles_list, net, method, ys=None, display=False, device='cpu'):
    explains = []
    th_min = float('inf')
    th_max = -float('inf')
    acts = []
    mols = []
    
    explainator = method(net) 
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        data = gen_graph(mol)
        attributions = explainator.attribute(data.x.unsqueeze(0).to(device),
                                                         #baselines=torch.zeros_like(data.x).unsqueeze(0).to(device),
                                                         additional_forward_args=(data.edge_index.unsqueeze(0).to(device),),
                                                         #method='gausslegendre',
                                                         #internal_batch_size=1,  #NEEDED in IG, OTHERWISE FUCKS UP
                                                       )
        explain = attributions.squeeze(0)
        batch = Batch.from_data_list([data]).to(device)
        out = net(batch.x, batch.edge_index, batch.batch)
        acts.append(out.item())
        mols.append(mol)
        explain = explain.mean(-1).detach().cpu().numpy()
        explains.append(explain)
        th_min = min(np.min(explain), th_min)
        th_max = max(np.max(explain), th_max)

    
    vmin = th_min
    vmax = th_max

    l = []
    for explain, mol in zip(explains, mols):
        explain = (explain - vmin)/(vmax - vmin) * 100

        atoms = explain
        highlightMap = {}
        for i, score in enumerate(atoms):
            r, b = 0, 0
            if score > 50:
                r = (score-50)/50
            else:
                b = score/50

            highlightMap[i] = [r, 0, b]
        #print(highlightMap)
        l.append(Draw.MolToImage(mol, size=(300, 300), highlightMap=highlightMap))
    
    if display:
        for i, (elem, act) in enumerate(zip(l, acts)):
            print(act)
            if ys:
                print(ys[i])
            display(elem)
    
    return list(zip(l, acts))