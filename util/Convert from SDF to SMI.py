from rdkit.Chem.PandasTools import LoadSDF
df = LoadSDF('D:\AIRO\RL\Project\dataset\B-raf.sdf', smilesName='SMILES')
SMILES = df['SMILES']
with open('B-raf_dataset.smi', 'w') as f:
    for row in SMILES:
        f.write(row + '\n')

df = LoadSDF('D:\AIRO\RL\Project\dataset\Aurora-A.sdf', smilesName='SMILES')
SMILES = df['SMILES']
with open('Aurora-A_dataset.smi', 'w') as f:
    for row in SMILES:
        f.write(row + '\n')

