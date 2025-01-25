import numpy as np
from typing import List

from Vocabulary.vocabulary import Vocabulary, Tokenizer


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


def vocabulary_from_SMILES(path: str | List[str]):
        '''
        Creates a vocabulary object from List of tokenized SMILES
        Params: path: path or list of paths to SMILES file/s
        :param path: path to 

        '''
        vocabulary = Vocabulary()
        SMILES_array = []
        if type(path) == list:
            for p in path: 
                SMILES_array += readSMILES(p)
        tokenizer = Tokenizer()
        for i, s in enumerate(SMILES_array):
            SMILES_array[i] = tokenizer.tokenize(s)
        
        s = list(map(set, SMILES_array))
        res = set()
        for se in s:
            res = res.union(se)
        #print(res)
        res = sorted(res)
        if '^' in res:
            res.remove('^')
        if '$' in res:
            res.remove('$')
        for s in res:
            vocabulary.add(s)
        
        return vocabulary

if __name__ == '__main__':
    print(len(readSMILES('data\Aurora-A_dataset.smi')))



