import numpy as np


def readSMILES(path: str):
    '''
    Reads a \n separated SMILES file
    Params:
    :param path: (str) path to SMILES file (.smi)
    '''
    assert path.endswith('.smi')

    with open(path, 'r') as SMILES:
        SMILES_file = SMILES.read()
        SMILES_list = SMILES_file.split(sep='\n')
    
    while '' in SMILES_list:
        SMILES_list.remove('')
    return np.array(SMILES_list)

if __name__ == '__main__':
    print(len(readSMILES('data\Aurora-A_dataset.smi')))

