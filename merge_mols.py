from rdkit import Chem
def merge_mols(m1, m2, idx1, idx2, bond_type):
	try:
		newmol = Chem.RWMol(m1)
		added = {}
		for i, a in enumerate(m2.GetAtoms()):
			newmol.AddAtom(a)
			added[i] = newmol.GetNumAtoms() - 1
		newmol.AddBond(idx1, added[idx2], bond_type)
		for b in m2.GetBonds():
			newmol.AddBond(added[b.GetBeginAtomIdx()], added[b.GetEndAtomIdx()],b.GetBondType())
		return Chem.MolFromSmiles(Chem.MolToSmiles(newmol))
	except:
		return None


if __name__ == '__main__':
	m1 = Chem.MolFromSmiles('CCC')
	m2 = Chem.MolFromSmiles('N')
	b = Chem.rdchem.BondType.TRIPLE

	m1 = merge_mols(m1,m2, 0, 0, b)
	print(m1)
	m1 = merge_mols(m1,m2, 0, 0, b)
	print(m1)
	m1 = merge_mols(m1,m2, 0, 0, b)
	print(m1)
	#print(Chem.MolToSmiles(m1))
	from rdkit.Chem import Draw
	Draw.MolsToGridImage([m2])