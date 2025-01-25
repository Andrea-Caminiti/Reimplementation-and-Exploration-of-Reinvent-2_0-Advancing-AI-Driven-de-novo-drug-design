import numpy as np
import torch

from collections.abc import Mapping
from typing import List
import re

class TokenAlreadyInVocabError(Exception):
    pass
class IndexError(Exception):
    pass

class Vocabulary:
    '''
    Data structure for token storage and conversion
    '''

    def __init__(self, tokens: Mapping = None, starting_id: int = 2):
        '''
        Params: 
        :param tokens: (optional) (dict) Dictionary of token, id pairs
        :param starting_id: (optional) (int) Starting index for the new vocabulary
        '''

        self.tokensToId = {'$': 0, '^': 1}
        self.idToTokens = {0: '$', 1: '^'}
        self.current_id = starting_id

        if tokens:
            self.update(tokens)
            self._swap()
        
    def _add(self, token: str, id: int):
        '''
        Private method to add new token, id pair to the vocabulary
        Params: 
        :param token: (str) Token to add to the dictionary
        :param id: (int) Index for the new token
        '''
        if id in self.idToTokens:
            raise IndexError('Index already in the vocabulary')
        elif token in self.tokensToId:
            raise TokenAlreadyInVocabError("Token already in vocabulary")
        else: 
            self.tokensToId[token] = id
            self.idToTokens[id] = token
            
        
    def update(self, tokens: Mapping):
        '''
        Method to add multiple tokens at once
        Params: 
        :param tokens: (dict) Dictionary of token, id pairs
        '''
        for token, id in tokens.items():
            if type(token) != str or type(id) != int:
                raise ValueError('Wrong tokens dictionary format. Expected string, int pairs')
            try: 
                self._add(token, id)
            except IndexError: 
                self._add(token, self.current_id)
                self.current_id += 1
            except TokenAlreadyInVocabError: 
                pass
    
    def add(self, token: str):
        '''
        Method to add a single token
        Params: 
        :param token: (str) Token to add
        '''
        self._add(token, self.current_id)
        self.current_id += 1

    def encode(self, token: str):
        '''
        Method to get the id of a specific token if present in the vocabulary, else returns None
        Params:
        :param token: string corresponding to the token to encode
        '''
        return self.tokensToId.get(token, None)
    
    def encode_sequence(self, sequence: List[str]):
        '''
        Method to get the encoding of a specific sequence
        Params:
        :param sequence: List corresponding to the sequence to encode
        '''
        res = np.zeros_like(sequence, dtype=np.float32)
        for i, token in enumerate(sequence):
            res[i] = self.tokensToId.get(token, None)
            
        return res
    
    def decode(self, id: int | List[int]):
        '''
        Method to get the token of a specific id if present in the vocabulary, else returns None
        Params:
        :param id: integer correspondin to the id to decode
        '''
        if type(id) == int:
            return self.idToTokens.get(id, None)
        else:
            return np.array(list(map(self.idToTokens.get, id.tolist())))

    def __getitem__(self, x: int | str):
        if type(x) == int:
            return self.idToTokens[x]
        elif type(x) == str:
            return self.tokensToId[x]
        else: 
            raise ValueError('Input must be either integer or string')

    def __delitem__(self, x: str | int):
        if type(x) == int:
            token = self.idToTokens[x]
            del self.idToToken[x]
            del self.tokensToId[token]
        elif type(x) == str:
            id = self.tokensToId[x]
            del self.idToToken[id]
            del self.tokensToId[x]
        else: 
            raise ValueError('Input must be either integer or string')
        
    def __contains__(self, x: str | int):
        if type(x) == int:
            return x in self.idToTokens
        elif type(x) == str:
            return x in self.tokensToId
        else: 
            raise ValueError('Input must be either integer or string')
        
    def __eq__(self, voc):
        return self.tokensToId == voc.tokensToId and self.idToTokens == voc.idToTokens
    
    def __len__(self):
        return len(self.tokensToId)
    

class Tokenizer:
    '''
    Class to handle tokenization of SMILES strings
    '''
    def __init__(self):
        self.REGEXPS = {
        "brackets": re.compile(r"(\[[^\]]*\])"),
        "2_ring_nums": re.compile(r"(%\d{2})"),
        "brcl": re.compile(r"(Br|Cl)")
        }
        self.ORDER = ["brackets", "2_ring_nums", "brcl"]
    
    def tokenize(self, data, with_begin_and_end=True):
        """Tokenizes a SMILES string."""
        def split_by(data, regexps):
            if not regexps:
                return list(data)
            regexp = self.REGEXPS[regexps[0]]
            splitted = regexp.split(data)
            tokens = []
            for i, split in enumerate(splitted):
                if i % 2 == 0:
                    tokens += split_by(split, regexps[1:])
                else:
                    tokens.append(split)
            return tokens

        tokens = split_by(data, self.ORDER)
        if with_begin_and_end:
            tokens = ["^"] + tokens + ["$"]
        return tokens

    def untokenize(self, tokens):
        """Untokenizes a SMILES string."""
        smi = ""
        for token in tokens:
            if token == "$":
                break
            if token != "^":
                smi += token
        return smi
