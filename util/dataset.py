import numpy as np
from typing import List
import torch

from util.SMILES import readSMILES
from torch.utils.data import Dataset, DataLoader
from Vocabulary.vocabulary import Vocabulary, Tokenizer

class SMILESDataset(Dataset):
    '''
    Dataset for SMILES strings
    '''

    def __init__(self, path: str, vocabulary: Vocabulary, tokenizer: Tokenizer = Tokenizer()):
        '''
        Params:
        :param path: (str) path to file containing the SMILES strings 
        :param vocabulary: (Vocabulary.vocabulary.Vocabulary) vocabulary of the prior
        :param tokenizer: (Vocabulary.vocabulary.Tokenizer) tokenizer of the prior
        '''
        def pad(encoded_seqs: List):
            """
            Function to take a list of encoded sequences and turn them into equally long sequences
            Params:
            :param encoded_seqs:
            """
        
            max_length = max([seq.size for seq in encoded_seqs])
            padded_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                padded_arr[i, :seq.size] = torch.LongTensor(seq)
            return padded_arr
        
        self.SMILES_array = readSMILES(path)
        for i, s in enumerate(self.SMILES_array):
            self.SMILES_array[i] = tokenizer.tokenize(s)
        
        self.SMILES_array = list(map(vocabulary.encode_sequence, self.SMILES_array))
        self.SMILES_array = pad(self.SMILES_array)
        self.SMILES_array = torch.LongTensor(self.SMILES_array)
            
        self.train = self.SMILES_array[:int(0.66 * len(self.SMILES_array))]
        self.val = self.SMILES_array[int(0.66 * len(self.SMILES_array)):]





def create_dataloader(path: str, vocabulary: Vocabulary, tokenizer: Tokenizer = Tokenizer(), batch_size: int = 128, validation: bool = False):
    dset = SMILESDataset(path, vocabulary, tokenizer)
    if validation:
        train_dataloader = DataLoader(dset.train, batch_size=batch_size)
        val_dataloader = DataLoader(dset.val, batch_size=batch_size)

        return train_dataloader, val_dataloader
    return DataLoader(dset.SMILES_array, batch_size=batch_size)