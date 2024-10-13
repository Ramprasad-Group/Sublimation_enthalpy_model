import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

def main():
	# all compounds
	compound = pd.read_csv("./547smiles.csv")  #smiles input ***********edit
	compound = compound[['SMILES']].copy() #***********edit
	# compound = compound.drop_duplicates().reset_index()
	# descriptor list
	descList = [desc[0] for desc in Descriptors.descList]
	# add columns
	# compound['num_atoms'] = np.nan
	for i in range(len(descList)):
		compound[descList[i]] = np.nan
	# loop over the compound to obtain descriptors
	for i in range(compound.shape[0]):
		print(i)
		smiles = compound.loc[i,'SMILES']    #***********edit
		m = Chem.MolFromSmiles(smiles)
		if m is not None:
			for label, func in Descriptors.descList:
				compound.loc[i,label] = func(m)
			# compound.loc[i,'num_atoms'] = Chem.AddHs(m).GetNumAtoms()
	compound.to_csv('./588COD_fp.csv', index = False)   #***********edit


if __name__=='__main__':
	main()
