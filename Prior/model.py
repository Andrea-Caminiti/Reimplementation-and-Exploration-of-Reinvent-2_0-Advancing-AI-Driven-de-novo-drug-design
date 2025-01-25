import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn

from collections.abc import Mapping
from typing import List, Tuple

from tqdm import tqdm

from Vocabulary import vocabulary as vc 
from util.SMILES import readSMILES, vocabulary_from_SMILES

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
        super().__init__()
        #Embedding params
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # Actual RNN params
        self.cell_type = cell_type.lower()
        self.num_layers = num_layers
        self.layer_size = layer_size
        self.dropout = dropout

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        match cell_type: 
            case 'lstm': self.Rnn = nn.LSTM(self.embedding_dim, self.layer_size, self.num_layers, batch_first=True, dropout=self.dropout)
            case 'gru': self.Rnn = nn.GRU(self.embedding_dim, self.layer_size, self.num_layers, batch_first=True, dropout=self.dropout) 
            case _: raise ValueError('Invalid cell_type parameter')
        self.Linear = nn.Linear(self.layer_size, self.vocab_size)

    def forward(self, x: torch.Tensor, hidden: torch.Tensor = None):
        '''
        Forward pass of the model.
        Params:
        :param x: (torch.Tensor) Sequence to be passed through the network
        :param hidden: (torch.Tensor) Hidden state
        '''
        
        batch_size, seq_len = x.size()
        if not hidden: 
            size = (self.num_layers, batch_size, self.layer_size)
            match self.cell_type:
                case 'lstm': hidden = [torch.zeros(*size).to(device='cuda'), torch.zeros(*size).to(device='cuda')]
                case 'gru': hidden = torch.zeros(*size).to(device='cuda')
        
        emb = self.embedding(x)
        
        res, hidden_state = self.Rnn(emb, hidden)
        del emb, hidden
        
        res = self.Linear(res)

        return res, hidden_state
    
class Prior:

    def __init__(self, vocabulary: vc.Vocabulary = vc.Vocabulary(), tokenizer: vc.Tokenizer = vc.Tokenizer(), 
                RNN_params: Mapping = None, max_seq_length: int = 256,
                smiles_paths: str = ['data\Aurora-A_dataset.smi', 'data\B-raf_dataset.smi'],  
                use_cuda: bool = False ):
        '''
        Generative model to create new SMILES strings
        Params: 
        :param vocabulary: (model.vocabulary.Vocabulary) vocabulary of SMILES tokens
        :param tokenizer: (model.vocabulary.Tokenizer) tokenizer for SMILES strings
        :param RNN_params: (dict) dictionary of parameters for internal RNN network
        :param max_seq_length: (int) maximum sequence length per each input 
        :param use_cuda: (boolean) boolean to activate cuda computation or not
        '''
        self.vocabulary = vocabulary
        self.vocab_length = len(vocabulary)

        self.tokenizer = tokenizer
        
        self.max_seq_length = max_seq_length

        self.use_cuda = use_cuda

        if not RNN_params: 
            RNN_params = {}

        self.RNN_params = RNN_params

        if self.vocab_length == 2: 
            self.vocabulary = vocabulary_from_SMILES(smiles_paths)
            self.vocab_length = len(self.vocabulary)
            
        self.RNN = RNN(self.vocab_length, **self.RNN_params)
        if self.use_cuda:
            self.RNN.to('cuda')

        self.loss = nn.NLLLoss(reduction='none')

    def save(self, path):
        '''
        Save prior to file
        Params:
        :param path: (str) path to file where to save the model
        '''

        params = {
            'vocabulary': self.vocabulary,
            'tokenizer': self.tokenizer,
            'max_sequence_length': self.max_seq_length,
            'RNN_params': self.RNN_params(),
            'network': self.RNN.state_dict()
        }

        torch.save(params, path)

    def load_prior(self, path: str):
        '''
        Load prior from file
        Params:
        :param path: (str) path to file where the model is saved
        '''

        if torch.cuda.is_available():

            params = torch.load(path)

        else:
            params = torch.load(path, map_location=lambda storage, loc: storage)

        model = Prior(
            vocabulary=params['vocabulary'],
            tokenizer=params['tokenizer'],
            RNN_params=params['RNN_params'],
            max_sequence_length=params['max_sequence_length']
        )

        model.RNN.load_state_dict(params["network"])
        
        return model

    def likelihood(self, sequences: List):
        """
        Computes the likelihood of a given sequence (already encoded SMILES strings).
        Params:
        :param sequences: (torch.Tensor) A batch of sequences
        :return:  (torch.Tensor) Log likelihood for each example.
        """
        #print(sequences[:, :-1].size())
        logits, _ = self.RNN(sequences[:, :-1])
        log_probs = logits.log_softmax(dim=2)
        return self.loss(log_probs.transpose(1, 2), sequences[:, 1:]).sum(dim=1)
    
    def likelihood_smiles(self, smiles: List):
        '''
        Returns the likelihood of a given tensor of SMILES strings
        Params:
        :param smiles: (torch.Tensor) tensor of SMILES string'''
        tokens = [self.tokenizer.tokenize(smile) for smile in smiles]
        encoded = [self.vocabulary.encode(token) for token in tokens]
        sequences = [torch.tensor(encode, dtype=torch.long) for encode in encoded]

        def collate_fn(encoded_seqs: List):
            """
            Function to take a list of encoded sequences and turn them into a batch
            Params:
            :param encoded_seqs:
            """
            max_length = max([seq.size(0) for seq in encoded_seqs])
            collated_arr = torch.zeros(len(encoded_seqs), max_length, dtype=torch.long)  # padded with zeroes
            for i, seq in enumerate(encoded_seqs):
                collated_arr[i, :seq.size(0)] = seq
            return collated_arr

        padded_sequences = collate_fn(sequences)
        return self.likelihood(padded_sequences)

    def sample_smiles(self, num: int = 128, batch_size: int = 128):
        """
        Samples n SMILES from the model.
        Params:
        :param num: Number of SMILES to sample.
        :param batch_size: Number of sequences to sample at the same time.
        """
        batch_sizes = [batch_size for _ in range(num // batch_size)] + [num % batch_size]
        smiles_sampled = []
        likelihoods_sampled = []

        for size in batch_sizes:
            if not size:
                break
            seqs, likelihoods = self.sample(batch_size=size)
            smiles = [self.tokenizer.untokenize(self.vocabulary.decode(seq)) for seq in seqs.cpu().numpy()]

            smiles_sampled.extend(smiles)
            likelihoods_sampled.append(likelihoods.data.cpu().numpy())

            del seqs, likelihoods
        return smiles_sampled, np.concatenate(likelihoods_sampled)
    
    def sample(self, batch_size: int = 128):
        '''
        Samples batch_size SMILES from the prior
        Params:
        :param batch_size: how many SMILES string to sample
        '''

        in_vector = torch.ones(batch_size, dtype=torch.long, device='cuda')
        sequences = [torch.ones([batch_size, 1], dtype=torch.long, device='cuda')]
        hidden_state = None
        neg_log_like = torch.zeros(batch_size).to(device='cuda')

        for _ in tqdm(range(self.max_seq_length - 1), desc='Sampling...'):
            logits, hidden_state = self.RNN(in_vector.unsqueeze(1), hidden_state)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1).to(device='cuda')
            log_probs = logits.log_softmax(dim=1).to(device='cuda')
            in_vector = torch.multinomial(probabilities, 1).view(-1).to(device='cuda')
            sequences.append(in_vector.view(-1, 1))
            neg_log_like += self.loss(log_probs, in_vector)
            del log_probs, probabilities
            if in_vector.sum() == 0:
                break

        sequences = torch.cat(sequences, 1)
        return sequences.data, neg_log_like


        


