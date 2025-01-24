import numpy as np
from typing import List

from Vocabulary.vocabulary import Vocabulary


def readSMILES(path: str):
    '''
    Reads a \\n separated SMILES file
    Params:
    :param path: (str) path to SMILES file (.smi)
    '''
    assert path.endswith('.smi')

    with open(path, 'r') as SMILES:
        SMILES_file = SMILES.read()
        SMILES_list = SMILES_file.split(sep='\n')
    
    while '' in SMILES_list:
        SMILES_list.remove('')
    return SMILES_list


def vocabulary_from_SMILES_array(tokenized_SMILES: List[List[str]]):
        '''
        Creates a vocabulary object 
        '''
        vocabulary = Vocabulary()
        s = list(map(set, tokenized_SMILES))
        res = set()
        for se in s:
            res = res.union(se)
        res = sorted(res)
        res.remove('^')
        res.remove('$')
        for s in res:
            vocabulary.add(s)
        
        return vocabulary

if __name__ == '__main__':
    print(len(readSMILES('data\Aurora-A_dataset.smi')))



