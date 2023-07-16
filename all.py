# several input files in the same directory
# generate ligand based PH4 (chemical_features.pdb) using RDKit

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
import os
import glob

# get all sdf files in the directory
sdf_files = glob.glob('/content/drive/MyDrive/BackupForColab/*.sdf')

# print the names and number of all the sdf files
print(f'Total number of sdf files: {len(sdf_files)}')

# initialize a feature factory
fdefName = os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef')
factory = ChemicalFeatures.BuildFeatureFactory(fdefName)


with open('/content/drive/MyDrive/BackupForColab/chemical_features.pdb', 'w') as f:

    # process each sdf file
    for sdf_file in sdf_files:
        print(sdf_file)

        # load the molecule from the sdf file
        supplier = Chem.SDMolSupplier(sdf_file)
        mols = [x for x in supplier if x is not None]

        # for each molecule, compute the features and write them
        for i, mol in enumerate(mols):
            # for each atom in the molecule, set a custom property with the atom identifier
            for atom in mol.GetAtoms():
                atom.SetProp('my_id', f'{i}_{atom.GetIdx()}')

            feats = factory.GetFeaturesForMol(mol)
            chembl_id = mol.GetProp('_Name').split('|')[0]  # extract the ChEMBL ID
            for j, feat in enumerate(feats):
                x, y, z = feat.GetPos()
                # retrieve atom ID
                atom_id = f"{i}_{feat.GetAtomIds()[0]}"
                f.write(f"ATOM  {j+1:5d} {feat.GetFamily():^4} {feat.GetType():<3} {i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f} {chembl_id} {atom_id}\n")

    f.write('END\n')




import numpy as np
from scipy.spatial import distance
import csv
import pandas as pd
from rdkit.Chem import AllChem

# define a mapping between the feature types
feature_mapping_rdkit = {
    'LumpedHydrophobe': 'Hydrophobe',
}

feature_mapping_pyrod = {
    'hd2': 'Donor',
    'hd': 'Donor',
    'ha2': 'Acceptor',
    'ha': 'Acceptor',
    'pi': 'PosIonizable',
    'ai': 'Aromatic',
    #'hi': 'Hydrophobe',
    'hi': None,
    'pi': 'PosIonizable',
    'ni': 'NegIonizable',
    'ev': None,
}

# define the structure of a pharmacophore feature
class PharmacophoreFeature:
    def __init__(self, type, position, molecule_num, atom_num, tol, chembl_id=None, atom_type=None, ID=None):
        self.type = type
        self.position = position
        self.molecule_num = molecule_num
        self.atom_num = atom_num
        self.tol = tol
        self.chembl_id = chembl_id
        #self.feature_type = feature_type
        self.atom_type = atom_type
        self.ID = ID


# define a function to parse a PDB file and extract the pharmacophore features
def parse_pyrod_pdb(filename):
    pharmacophore = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                atom_num = int(parts[1])
                atom_type = parts[2]
                feature_type = parts[3]
                molecule_num = int(parts[5])
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
                tol = float(parts[9])
                # map the feature type to another feature type
                if feature_type == 'hda':
                    if atom_type == 'C':
                        # create two features for 'C'
                        pharmacophore.append(PharmacophoreFeature('Donor', np.array([x, y, z]), molecule_num, atom_num, tol, atom_type = atom_type))
                        pharmacophore.append(PharmacophoreFeature('Acceptor', np.array([x, y, z]), molecule_num, atom_num, tol, atom_type = atom_type))
                    elif atom_type == 'Pd':
                        pharmacophore.append(PharmacophoreFeature('Donor', np.array([x, y, z]), molecule_num, atom_num, tol, atom_type = atom_type))
                    elif atom_type == 'Pa':
                        pharmacophore.append(PharmacophoreFeature('Acceptor', np.array([x, y, z]), molecule_num, atom_num, tol, atom_type = atom_type))
                else:
                    mapped_type = feature_mapping_pyrod.get(feature_type, feature_type)
                    pharmacophore.append(PharmacophoreFeature(mapped_type, np.array([x, y, z]), molecule_num, atom_num, tol, atom_type = atom_type))
    return pharmacophore


def parse_rdkit_pdb(filename):
    pharmacophore = []
    with open(filename, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                parts = line.split()
                atom_num = int(parts[1])
                feature_type = parts[2]
                atom_type = parts[3]
                molecule_num = int(parts[4])
                x = float(parts[5])
                y = float(parts[6])
                z = float(parts[7])
                chembl_id = parts[8]
                ID = parts[9]
                # map the feature type to another feature type
                mapped_type = feature_mapping_rdkit.get(feature_type, feature_type)
                # create a pharmacophore feature
                feature = PharmacophoreFeature(mapped_type, np.array([x, y, z]), molecule_num, atom_num, None, chembl_id, atom_type = atom_type, ID=ID)
                pharmacophore.append(feature)
    return pharmacophore


# define a function to calculate the similarity score for a pair of features
def calculate_dist(feature1, feature2):
    # calculate the Euclidean distance between the features
    dist = distance.euclidean(feature1.position, feature2.position)
    return dist



#parse the PDB files
pharmacophore1 = parse_rdkit_pdb('/content/drive/MyDrive/BackupForColab/chemical_features.pdb')
pharmacophore2 = parse_pyrod_pdb('/content/drive/MyDrive/BackupForColab/aligned_super_pharmacophore.pdb')



def write_pdb_file(filename, pharmacophore):
    with open(filename, 'w') as f:
        for feature in pharmacophore:
            if feature.type is not None:
                x, y, z = feature.position
                chembl_id_str = f" {feature.chembl_id}" if feature.chembl_id is not None else ""
                atom_type_str = f" {feature.atom_type}" if feature.atom_type is not None else ""
                id_str = f" {feature.ID}" if feature.ID is not None else ""
                f.write(f"ATOM  {feature.atom_num:5d} {feature.type:^4} {feature.molecule_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{chembl_id_str}{atom_type_str}{id_str}\n")  # add the ID string to the line
        f.write('END\n')

# write the mapped PDB files
write_pdb_file('/content/drive/MyDrive/BackupForColab/mapped_chemical_features.pdb', pharmacophore1)
write_pdb_file('/content/drive/MyDrive/BackupForColab/mapped_aligned_super_pharmacophore.pdb', pharmacophore2)


# create empty list to store the scores
scores = []

# define specific cutoffs for 'P', 'Pa', and 'Pd'
cutoffs = {'P': 3.9, 'Pa': 3.9, 'Pd': 3.9}

# for each feature in the first pharmacophore
for feature1 in pharmacophore1:
    # for each feature in the second pharmacophore
    for feature2 in pharmacophore2:
        # check whether the features match
        if feature1.type == feature2.type:
            # calculate the similarity score for this pair of features
            dist = calculate_dist(feature1, feature2)

            # check the atom type and compare the score with the appropriate cutoff
            if feature1.type == 'Aromatic':
                if feature2.atom_type == 'C':
                    if dist > feature2.tol:
                        dist = 0
                elif feature2.atom_type == 'P':
                    continue  # skip this iteration if atom type is 'P' and molecule type is 'Aromatic'
            else:
                if feature2.atom_type in ['P', 'Pa', 'Pd']:
                    if dist > cutoffs[feature2.atom_type]:
                        dist = 0
                elif feature2.atom_type == 'C':
                    if dist > feature2.tol:
                        dist = 0

            # calculate 1/score if score is not 0
            inv_dist = 1/dist if dist != 0 else 0

            # append score along with feature types and atom/molecule numbers to scores list
            scores.append([feature1.molecule_num, feature1.atom_num, feature1.type, feature1.chembl_id, feature1.ID, feature2.molecule_num, feature2.atom_num, feature2.type, inv_dist])



# convert these lists into pandas DataFrames
df1 = pd.DataFrame([{'type': f.type, 'molecule_num': f.molecule_num, 'atom_num': f.atom_num, 'position': f.position, 'chembl_id': f.chembl_id, 'ID': f.ID} for f in pharmacophore1])
df2 = pd.DataFrame([{'type': f.type, 'molecule_num': f.molecule_num, 'atom_num': f.atom_num, 'position': f.position, 'tol': f.tol, 'chembl_id': f.chembl_id} for f in pharmacophore2])

# find common types in both dataframes
common_types = set(df1['type']).intersection(df2['type'])

print(common_types)


# convert the scores list into a pandas DataFrame
df_scores = pd.DataFrame(scores, columns=['Molecule_Num_RDKit', 'Atom_Num_RDKit', 'Type_RDKit', 'ChEMBL_ID_RDKit', 'Atom_ID_RDKit', 'Molecule_Num_PyRod', 'Atom_Num_PyRod', 'Type_PyRod', 'Score'])
# print the DataFrame to check the result
#df_scores



# Group the dataframe by 'ChEMBL_ID_RDKit' and 'Molecule_Num_RDKit',
# and only keep the row with the maximum 'Score' in each group
df_scores_group = df_scores.groupby(['ChEMBL_ID_RDKit', 'Molecule_Num_RDKit']).max().reset_index()

# Print the grouped dataframe
df_scores_group



# Load the csv file
df_ic50 = pd.read_csv('/content/drive/MyDrive/BackupForColab/IC50_pChembl_value_preprocessed.csv')

# Select only needed columns and rename 'Molecule ChEMBL ID' to 'ChEMBL_ID_RDKit'
df_ic50 = df_ic50[['Molecule ChEMBL ID', 'pChEMBL Value']].rename(columns={'Molecule ChEMBL ID': 'ChEMBL_ID_RDKit'})

# Merge on 'ChEMBL_ID_RDKit'
df_scores_group_act = pd.merge(df_scores_group, df_ic50, on='ChEMBL_ID_RDKit', how='left')

# Create 'activity_classification' column
df_scores_group_act['activity_classification'] = df_scores_group_act['pChEMBL Value'].apply(lambda x: 'active' if x >= 6 else 'inactive')



# define the maximum number of rows per file
max_rows_per_file = 1000000

# calculate the number of files needed
num_files = len(df_scores_group_act) // max_rows_per_file + 1

# split the DataFrame into chunks and write each chunk to a separate file
for i in range(num_files):
    df_chunk = df_scores_group_act[i*max_rows_per_file:(i+1)*max_rows_per_file]
    df_chunk.to_csv(f'/content/drive/MyDrive/BackupForColab/scores_{i+1}.csv', index=False)



# Create a new column that combines the Type and Molecule number
df_scores_group_act['Type_Molecule'] = df_scores_group_act['Type_PyRod'] + '_' + df_scores_group_act['Molecule_Num_PyRod'].astype(str)

# First, split the 'Atom_ID_RDKit' to get the molecule index 'i'
df_scores_group_act['i'] = df_scores_group_act['Atom_ID_RDKit'].apply(lambda x: int(x.split('_')[0]))

# Add a new column that combines 'ChEMBL_ID_RDKit' and 'Molecule_Num_RDKit'
#df_scores_group_act['ChEMBL_Mol_RDKit'] = df_scores_group_act['ChEMBL_ID_RDKit'] + '_' + df_scores_group_act['Atom_ID_RDKit'].astype(str)
df_scores_group_act['ChEMBL_Mol_RDKit'] = df_scores_group_act['ChEMBL_ID_RDKit'] + '_' + df_scores_group_act['i'].astype(str)


# Pivot the DataFrame to create a new column for each Type and Molecule number
#df_pivot = df_scores_group.pivot_table(index='ChEMBL_Mol_RDKit', columns='Type_Molecule', values='Score', aggfunc=lambda x: max([0.0] + [xi for xi in x if xi != 0]))
df_pivot = df_scores_group_act.pivot_table(index='ChEMBL_Mol_RDKit', columns='Type_Molecule', values='Score')

# Fill NaN values with 0.0
df_pivot.fillna(0.0, inplace=True)

# Reset the index to make the 'ChEMBL_Mol_RDKit' a column again
df_pivot.reset_index(inplace=True)


df_pivot.to_csv('/content/drive/MyDrive/BackupForColab/scores_feature_pivot_original.csv')



from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
import pandas as pd
import os
import glob
from numba import njit
import math

# load the PDB file
pdb_mol = Chem.MolFromPDBFile('/content/drive/MyDrive/BackupForColab/aligned_super_pharmacophore.pdb')

# get all sdf files in the directory
sdf_files = glob.glob('/content/drive/MyDrive/BackupForColab/*.sdf')
#sdf_files = glob.glob('/content/drive/MyDrive/BackupForColab/all_3D_washed_1.sdf')

# print the names and number of all the sdf files
print(f'Total number of sdf files: {len(sdf_files)}')

# parse the PDB file to get the 'ev' atoms and their 'tol' values
pdb_ev_atoms = []
pdb_ev_tols = []
pdb_types = []
pdb_molecule_nums = []

with open('/content/drive/MyDrive/BackupForColab/aligned_super_pharmacophore.pdb', 'r') as f:
    for line in f:
        if line.startswith('ATOM'):
            parts = line.split()
            feature_type = parts[3]
            if feature_type == 'ev':
                x = float(parts[6])
                y = float(parts[7])
                z = float(parts[8])
                tol = float(parts[9])
                pdb_ev_atoms.append(np.array([x, y, z]))
                pdb_ev_tols.append(tol)
                pdb_types.append(feature_type)
                pdb_molecule_nums.append(int(parts[5]))

pdb_ev_atoms = np.array(pdb_ev_atoms)
pdb_ev_tols = np.array(pdb_ev_tols)

# list to store the DataFrames for each SDF file
dfs = []

@njit
def calculate_scores_ev(pdb_ev_atoms, pdb_ev_tols, sdf_pos):
    scores_ev = []
    for i in range(len(pdb_ev_atoms)):
        pdb_pos = pdb_ev_atoms[i]
        tol = pdb_ev_tols[i]
        distance_ev = math.sqrt((pdb_pos[0] - sdf_pos[0])**2 + (pdb_pos[1] - sdf_pos[1])**2 + (pdb_pos[2] - sdf_pos[2])**2)
        if distance_ev < tol:
            score_ev = -1 / distance_ev
        else:
            score_ev = 0

        scores_ev.append(score_ev)
    return scores_ev

# process each sdf file
for sdf_file in sdf_files:
    print(sdf_file)

    # load the SDF file
    sdf_supplier = Chem.SDMolSupplier(sdf_file)
    sdf_mols = [x for x in sdf_supplier if x is not None]

    # initialize a counter for the total number of atoms
    total_atoms = 0

    # iterate over all molecules in the SDF file
    for i, sdf_mol in enumerate(sdf_mols):
        # add the number of atoms in the current molecule to the total
        total_atoms += sdf_mol.GetNumAtoms()

        # calculate and save the distances
        conf = sdf_mol.GetConformer()

        chembl_id = sdf_mol.GetProp('_Name').split('|')[0]  # extract the ChEMBL ID

        for sdf_atom in sdf_mol.GetAtoms():
            sdf_pos = np.array(conf.GetAtomPosition(sdf_atom.GetIdx()))
            scores_ev = calculate_scores_ev(pdb_ev_atoms, pdb_ev_tols, sdf_pos)

            # generate atom_id
            atom_id = f"{i}_{sdf_atom.GetIdx()}"

            # save the scores, atom_id, Chembl ID, Type_PyRod, Molecule_Num_PyRod as a DataFrame and add it to the list
            df_ev = pd.DataFrame({'ChEMBL_ID_RDKit': [chembl_id]*len(scores_ev),
                                  'Atom_ID_RDKit': [atom_id]*len(scores_ev),
                                  'Type_PyRod': pdb_types,
                                  'Molecule_Num_PyRod': pdb_molecule_nums,
                                  'Score_ev': scores_ev})
            dfs.append(df_ev)

    # print the total number of atoms in the SDF file
    print(f'Total number of atoms in {sdf_file}: {total_atoms}')




# concatenate all the DataFrames
df_all_ev = pd.concat(dfs, ignore_index=True)

# print the DataFrame
df_all_ev


# concatenate all the DataFrames
df_all_ev = pd.concat(dfs, ignore_index=True)

# First, split the 'Atom_ID_RDKit' to get the molecule index 'i'
df_all_ev['i'] = df_all_ev['Atom_ID_RDKit'].apply(lambda x: int(x.split('_')[0]))

# Find the index of the minimum 'Score_ev' for each group
idx = df_all_ev.groupby(['ChEMBL_ID_RDKit', 'i'])['Score_ev'].idxmin()

# Use this index to subset the original dataframe
df_all_ev_group = df_all_ev.loc[idx]

# Print the resulting DataFrame
df_all_ev_group



df_all_ev_group.to_csv('/content/drive/MyDrive/BackupForColab/scores_ev_group.csv')


# Load the csv file
df_ic50 = pd.read_csv('/content/drive/MyDrive/BackupForColab/IC50_pChembl_value_preprocessed.csv')

# Select only needed columns and rename 'Molecule ChEMBL ID' to 'ChEMBL_ID_RDKit'
df_ic50 = df_ic50[['Molecule ChEMBL ID', 'pChEMBL Value']].rename(columns={'Molecule ChEMBL ID': 'ChEMBL_ID_RDKit'})

# Merge on 'ChEMBL_ID_RDKit'
df_all_ev_group_act = pd.merge(df_all_ev_group, df_ic50, on='ChEMBL_ID_RDKit', how='left')

# Create 'activity_classification' column
df_all_ev_group_act['activity_classification'] = df_all_ev_group_act['pChEMBL Value'].apply(lambda x: 'active' if x >= 6 else 'inactive')



# Create a new column that combines 'ChEMBL_ID' and 'Atom_ID'
#df_all_ev_group_act = df_all_ev_group_act.assign(ChEMBL_Mol_RDKit = df_all_ev_group_act['ChEMBL_ID_RDKit'] + '_' + df_all_ev_group_act['Atom_ID_RDKit'].astype(str))
df_all_ev_group_act = df_all_ev_group_act.assign(ChEMBL_Mol_RDKit = df_all_ev_group_act['ChEMBL_ID_RDKit'] + '_' + df_all_ev_group_act['i'].astype(str))

# Create a new column that combines the Type and Molecule number
df_all_ev_group_act = df_all_ev_group_act.assign(Type_Molecule = df_all_ev_group_act['Type_PyRod'] + '_' + df_all_ev_group_act['Molecule_Num_PyRod'].astype(str))

# Create a pivot table from the df_all_ev DataFrame
#df_ev_pivot = df_all_ev_group.pivot_table(index='ChEMBL_Mol_RDKit', columns='Type_Molecule', values='Score_ev', aggfunc=lambda x: max([0.0] + [xi for xi in x if xi != 0]))
df_ev_pivot = df_all_ev_group_act.pivot_table(index='ChEMBL_Mol_RDKit', columns='Type_Molecule', values='Score_ev')

# Fill NaN values with 0.0
df_ev_pivot.fillna(0.0, inplace=True)

# Reset the index to make the 'ChEMBL_ID' a column again
df_ev_pivot.reset_index(inplace=True)



df_merged = pd.merge(df_pivot, df_ev_pivot, on='ChEMBL_Mol_RDKit', how='outer')
df_merged.fillna(0.0, inplace=True)

df_merged['ChEMBL_ID_RDKit'] = df_merged['ChEMBL_Mol_RDKit'].apply(lambda x: x.split('_')[0])
df_ic50['activity_classification'] = df_ic50['pChEMBL Value'].apply(lambda x: 'active' if x >= 6 else 'inactive')
df_merged = pd.merge(df_merged, df_ic50[['ChEMBL_ID_RDKit', 'activity_classification']], left_on='ChEMBL_ID_RDKit', right_on='ChEMBL_ID_RDKit', how='left')
# Drop 'ChEMBL_ID_RDKit' column from df_pivot
df_merged.drop('ChEMBL_ID_RDKit', axis=1, inplace=True)


df_merged.to_csv('/content/drive/MyDrive/BackupForColab/scores_pivot_merged.csv')
