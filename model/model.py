import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn

import vocabulary as vc

class RNN(nn.Module):
    '''
    RNN class to create the base for the model "Prior" that will have 
    either GRU or LSTM cells.
    '''

    def __init__(self, vocab_size:int, embedding_dim: int = 256, 
                 cell_type: str = 'lstm', num_layers: int = 5, layer_size: int = 512, dropout: float = 0.2
                ):
        '''
        Params:
        :param vocab_size: (int) Size of the vocabulary and thus the final linear output layer
        :param embedding_dim: (int) Dimension of the output of the Embedding layer
        :param cell_type: (str) Type of the cells, can be either "gru" or "lstm"
        :param num_layers: (int) number of RNN layers
        :param layer_size: (int) Dimension of each layer
        :param dropout: (float) Dropout to be used between layers
        '''
        
        #Embedding params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Actual RNN params
        self.cell_type = cell_type.lower()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.dropout = dropout

        self.Embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        match cell_type: 
            case 'lstm': self.Rnn = nn.LSTM(self.embedding_dim, self.layer_size, self.num_layers, batch_first=True, dropout=self.dropout)
            case 'gru': self.Rnn = nn.GRU(self.embedding_dim, self.layer_size, self.num_layers, batch_first=True, dropout=self.dropout) 
            case _: raise ValueError('Invalid cell_type parameter')
        self.Linear = nn.Linear(self.layer_size, self.vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        '''
        Forward pass of the model.
        Params:
        :param x: Sequence to be passed through the network
        :param hidden: Hidden state
        '''

        batch_size, seq_len = x.size()
        if not hidden: 
            size = (self._num_layers, batch_size, self._layer_size)
            match self.cell_type:
                case 'lstm': hidden = [torch.zeros(*size), torch.zeros(*size)]
                case 'gru': hidden = torch.zeros(*size)
            
        emb = self.Embedding(x)
        res, hidden_state = self.Rnn(emb, hidden)
        res = self.Linear(res)

        return res, hidden_state
    
class Prior:
    '''
    Generative model to create new SMILES strings
    '''

    def __init__(self, vocabulary: vc.Vocabulary):

        pass