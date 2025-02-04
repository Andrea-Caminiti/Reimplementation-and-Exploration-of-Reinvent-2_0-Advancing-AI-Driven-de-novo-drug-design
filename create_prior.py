from Prior.model import Prior
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH as general

from torch.utils.data import Dataset, DataLoader
from Vocabulary.vocabulary import Vocabulary, Tokenizer
from typing import List
from tqdm import tqdm 
from os.path import basename, join
import numpy as np
import torch
import scipy.stats as sps
import gc


def readSMILES(path: str, i: int):
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
    return SMILES_list[1000*i: 1000*(i+1)]

def vocabulary_from_SMILES(path: str, vocabulary: Vocabulary = Vocabulary()):
        '''
        Creates a vocabulary object from List of tokenized SMILES
        Params: path: path or list of paths to SMILES file/s
        :param path: path to 

        '''
        with open(path, 'r') as SMILES:
            SMILES_file = SMILES.read()
            SMILES_array = SMILES_file.split(sep='\n')
        tokenizer = Tokenizer()
        for i, s in enumerate(SMILES_array):
            SMILES_array[i] = tokenizer.tokenize(s)
        
        s = list(map(set, SMILES_array))
        res = set()
        for se in s:
            res = res.union(se)
        res = sorted(res)
        if '^' in res:
            res.remove('^')
        if '$' in res:
            res.remove('$')
        for s in res:
            if s not in vocabulary:
                vocabulary.add(s)
        
        return vocabulary

class SMILESDataset(Dataset):
    '''
    Dataset for SMILES strings
    '''

    def __init__(self, i: int, path: str, vocabulary: Vocabulary, tokenizer: Tokenizer = Tokenizer()):
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
        
        self.SMILES_array = readSMILES(path, i)
        for i, s in enumerate(self.SMILES_array):
            self.SMILES_array[i] = tokenizer.tokenize(s)
        
        self.SMILES_array = list(map(vocabulary.encode_sequence, self.SMILES_array))
        self.SMILES_array = pad(self.SMILES_array)
        self.SMILES_array = torch.LongTensor(self.SMILES_array).to('cuda')
            
        self.train = self.SMILES_array[:int(0.66 * len(self.SMILES_array))]
        self.val = self.SMILES_array[int(0.66 * len(self.SMILES_array)):]


def create_dataloader(i: int, path: str, vocabulary: Vocabulary, tokenizer: Tokenizer = Tokenizer(), batch_size: int = 64):
    
    vocabulary = vocabulary_from_SMILES(path, vocabulary)
    dset = SMILESDataset(i, path, vocabulary, tokenizer)    
    dloader = DataLoader(dset.SMILES_array, batch_size=batch_size)
    del dset
    return dloader, vocabulary

class LearningRate:
    '''
    Class to manipulate learning rate
    '''
    
    def __init__(self, 
                 prior: Prior = None, prior_path: str = None,
                 mode: str = 'constant',
                 max_v: float =  5e-3, min_v: float = 1e-5, 
                 step: int = 1, decay: float = 0.8, sample_size: int = 128,
                 patience: int = 5, validation: bool = False):
        '''
        Params:
        :param prior: (model.Prior) (optional) prior that will be trained
        :param prior_path: (str) (optional) path to load prior from file \n
        One between prior and prior_path must be given! \n
        :param mode: (str) (optional) Mode for the scheduler, possible values are: exponential, adaptive, and constant
        :param max_v: (float) (optional) Starting value for the learning rate
        :param min_v: (float) (optional) Final value for the learning rate
        :param step: (int) (optional) How many epochs to apply decay        
        :param decay: (float) (optional) decay for the learning rate, aka gamma
        :param sample_size: (int) (optional) amount of SMILES to sample
        :param patience: (int) (optional) how many epoch before applying decay with adaptive scheduler
        '''

       

class Trainer: 
    '''
    Trainer for a model
    '''

    def __init__(self, 
                 epochs: int, lr: LearningRate, bs: int,
                 save_path: str, save_epochs: int,  
                 smiles_path: str,
                 prior: Prior = None, prior_path: str = None,
                 starting_epoch: int = 1, early_stop: bool = False, patience: int = None,
                 mode: str = 'constant',
                 max_v: float =  5e-3, min_v: float = 1e-5, 
                 step: int = 1, decay: float = 0.8, sample_size: int = 128,
                 patience_lr: int = 5, validation: bool = False):
        
        '''
        Params:
        :param epochs: (int) epochs to train for
        :param lr: (LearningRate) LearningRate object to manage learning rate
        :param bs: (int) batch_size
        :param patience: (int) patience for EarlyStopping
        :param save_path: (str) path to save the trained model to
        :param save_epochs: (int) epochs beetween saving
        :param smiles_path: (str) path to smile dataset
        :param prior: (model.Prior) (optional) prior that will be trained
        :param prior_path: (str) (optional) path to load prior from file
        :param starting_epoch: (int) (optional) epoch from when training is starting \n
        One between prior and prior_path must be given! \n
        '''
        
        assert (prior != None or prior_path != None)
        assert (not early_stop and not patience) or (early_stop and patience != None)

        self.epochs = epochs
        self.batch_size = bs
        
        self.early_stop = early_stop
        self.patience = patience
        self.losses = []

        self.save_path = save_path
        self.save_epochs = max(save_epochs, 1)

        self.smiles_path = smiles_path

        self.prior = prior
        if prior_path:
            self.prior = Prior().load_prior(prior_path)
            self.prior.Rnn = self.prior.Rnn.cuda()
        
        self.starting_epoch = max(starting_epoch, 1)

        self.mode = mode.lower()
        self.value = max_v
        self.max_value = max_v
        self.min_value = min_v
        self.step = step
        self.decay = decay
        self.sample_size = sample_size
        self.patience_lr = patience_lr
        self.prior = prior
        self.validation = validation
        if self.mode == 'adaptive':
            self.metrics = []
        
        self.optimizer = torch.optim.Adam(self.prior.Rnn.parameters(), lr = self.max_value)

        self.scheduler = None

        match self.mode: 
            case 'exponential': \
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step, gamma=self.decay)

            case 'adaptive': \
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.decay, patience=self.patience_lr)

            case _ : pass
        
    def update(self, epoch: int):
        '''
        Updates the learning rate
        Params:
        :param epoch: (int) current epoch 
        '''
        
        match self.mode:
            case 'exponential': 
                self.scheduler.step(epoch)
                self.value = self.scheduler.get_last_lr()
            
            case 'adaptive': 
                m = np.mean(self.metrics[-6:]) # Use mean of the last 5 metrics
                self.scheduler.step(m, epoch)
                self.value = self.scheduler.get_last_lr()

            case _: pass

        if self.value < self.min_value:
            self.value = self.min_value
            self.mode = 'constant'

    def current_lr(self):
        return self.value
    
    def _add_metric(self, smiles_path: str, train_d_loader):
        def jsd(dists):
                num_dists = len(dists)
                avg_dist = np.sum(dists, axis=0) / num_dists
                return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists
        _, sampled_nlls = self.prior.sample_smiles(num=self.sample_size)
        if self.validation:
            training_nlls, valid_nlls = self._nlls(smiles_path, self.validation)
            self.metrics.append(jsd([sampled_nlls, training_nlls, valid_nlls]))
        else:
            training_nlls = self._nlls(smiles_path, train_d_loader)
            self.metrics.append(jsd([sampled_nlls, training_nlls]))
    
    def _nlls(self, path: str, train_d_loader, validation: bool = False):
        def pad(nlls):
            length = len(nlls)
            neg_lls = nlls
            if length < self.sample_size:
                delta = self.sample_size - length
                padding = []
                counter = 0
                
                for i in range(delta):
                    padding.append(nlls[counter])
                    if length == (counter + 1):
                        counter = 0
                    else:
                        counter += 1

                neg_lls = np.concatenate([nlls, padding])
            elif length > self.sample_size:
                neg_lls = np.random.choice(neg_lls, self.sample_size, replace=False)

            return neg_lls
        
    
        train = train_d_loader
        train_nll = []
        for batch in train:
            train_nll.append(self.prior.likelihood(batch))
        
        return pad(torch.concatenate(train_nll).cpu().detach().numpy())

    def train(self):
        end = self.epochs + self.starting_epoch - 1
        p = 0
        best_loss = 0.0
        for epoch in tqdm(range(self.starting_epoch, end), total = self.epochs, desc='Epochs'):
            for i in range(20):
                train_d_loader, self.prior.vocabulary = create_dataloader(i, self.smiles_path, self.prior.vocabulary, batch_size=self.batch_size )
                self.training_step(i, epoch, train_d_loader)
                del train_d_loader
                gc.collect()
                self.prior.save(f'{self.save_path}.{basename(self.smiles_path)[:-4]}')
        return 
    
    def training_step(self,i:int,  epoch: int, train_d_loader: DataLoader):
        
        for batch in tqdm(train_d_loader, total=len(train_d_loader), desc=f'Batch {i}', leave=False):
            batch = batch.long()
            loss = self.prior.likelihood(batch).mean()
            del batch
            self.scheduler.optimizer.zero_grad()
            self.optimizer.zero_grad()
            loss.backward()
            self.scheduler.optimizer.step()
            self.optimizer.step()
        self._add_metric(self.smiles_path, train_d_loader)
        if epoch % self.save_epochs == 0:
            data = basename(self.smiles_path)
            self.prior.save(f'{self.save_path}.{data[:-4]}.{epoch}.batch.{i}')
        
        gc.collect()
        return


if __name__ == '__main__':

    prior = Prior(use_cuda=True)
    prior.vocabulary = vocabulary_from_SMILES('data\dataset.smi')
    prior.save('priors\prior.dataset')
    
    trainer = Trainer(epochs=general['epochs'],
                      lr=LearningRate(),
                      bs=general['bs'],
                      early_stop=general['early_stop'],
                      patience=general['patience_train'],
                      save_path=general['save_path'],
                      save_epochs=general['save_epochs'],
                      smiles_path=general['smiles_path'],
                      prior=prior, 
                      prior_path=general['prior_path'],
                      starting_epoch=general['starting_epoch'],
                      mode=general['mode'],
                      max_v=general['max_v'], 
                      min_v=general['min_v'], 
                      step=general['step'], 
                      decay=general['decay'], 
                      sample_size=general['sample_size'],
                      patience_lr=general['patience_lr'], 
                      validation=general['validation']
                      ) 
    
    trainer.train()
    prior.save('priors/RandomPrior')