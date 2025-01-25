import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as fnn
import scipy.stats as sps
from tqdm import tqdm
from os.path import basename
import gc

from Prior.model import Prior
from util import SMILES
from util.dataset import SMILESDataset, create_dataloader

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
        assert (prior != None or prior_path != None) and not (prior and prior_path)

        self.mode = mode.lower()
        self.value = max_v
        self.max_value = max_v
        self.min_value = min_v
        self.step = step
        self.decay = decay
        self.sample_size = sample_size
        self.patience = patience
        self.prior = prior
        self.validation = validation
        if self.mode == 'adaptive':
            self.metrics = []

        if prior_path: 
            self.prior = Prior().load_prior(prior_path)
        
        self.optimizer = torch.optim.Adam(self.prior.RNN.parameters(), lr = self.max_value)

        self.scheduler = None

        match self.mode: 
            case 'exponential': \
                self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.step, gamma=self.decay)

            case 'adaptive': \
                self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.decay, patience=self.patience)

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
    
    def _add_metric(self, smiles_path: str):
        def jsd(dists):
                num_dists = len(dists)
                avg_dist = np.sum(dists, axis=0) / num_dists
                return np.sum([sps.entropy(dist, avg_dist) for dist in dists]) / num_dists
        
        _, sampled_nlls = self.prior.sample_smiles(num=self.sample_size)
        if self.validation:
            training_nlls, valid_nlls = self._nlls(smiles_path, self.validation)
        else:
            training_nlls = self._nlls(smiles_path)
        self.metrics.append(jsd([sampled_nlls, training_nlls, valid_nlls]))
    
    def _nlls(self, path: str, validation: bool = False):
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

            return neg_lls
        
        if validation: 
            train, val = create_dataloader(path, self.prior.vocabulary, self.prior.tokenizer, validation=validation)
            train_nll = []
            for batch in train:
                batch = batch.to(device='cuda')
                train_nll += self.prior.likelihood(batch)
            val_nll = []
            for batch in val:
                batch = batch.to(device='cuda')
                val_nll += self.prior.likelihood(batch)
            
            return pad(torch.concatenate(train_nll)), pad(torch.concatenate(val_nll))
        else:
            train = create_dataloader(path, self.prior.vocabulary, self.prior.tokenizer)
            train_nll = []
            for batch in train:
                batch = batch.to(device='cuda')
                train_nll += self.prior.likelihood(batch)
            
            return pad(torch.concatenate(train_nll))



class Trainer: 
    '''
    Trainer for a model
    '''

    def __init__(self, 
                 epochs: int, lr: LearningRate, bs: int,
                 save_path: str, save_epochs: int,  
                 smiles_path: str,
                 prior: Prior = None, prior_path: str = None,
                 starting_epoch: int = 1, early_stop: bool = False, patience: int = None):
        
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
        
        assert (prior != None or prior_path != None) and not (prior and prior_path)
        assert (not early_stop and not patience) or (early_stop and patience != None)

        self.epochs = epochs
        self.scheduler = lr
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
        
        self.starting_epoch = max(starting_epoch, 1)

    def train(self):
        end = self.epochs + self.starting_epoch - 1
        p = 0
        best_loss = 0.0
        for epoch in tqdm(range(self.starting_epoch, end), total = self.epochs, desc='Epochs'):
            gc.collect()
            epoch_loss = self.training_step(epoch)

            if self.early_stop:
                self.losses.append(epoch_loss)
                if len(self.losses) > 1 and p < self.patience and  best_loss - epoch_loss  < 10e-3:
                    best_loss = epoch_loss
                    p = 0
                elif epoch_loss > best_loss:
                    p += 1
                elif len(self.losses) == 0:
                    best_loss = epoch_loss

                if p >= self.patience:
                    data = basename(self.smiles_path)
                    self.prior.save(f'{self.save_path}.{data[:-4]}.{epoch}' )
        return 
    
    def training_step(self, epoch: int):
        train_d_loader = create_dataloader(self.smiles_path, self.prior.vocabulary, batch_size=self.batch_size)
        ls = [] # losses for every batch
        for batch in tqdm(train_d_loader, total=len(train_d_loader), desc=f'Epoch {epoch}', leave=False):
            gc.collect()
            batch = batch.long()
            if self.prior.use_cuda:
                batch = batch.to('cuda')
            #print(batch.size())
            loss = self.prior.likelihood(batch).mean()
            self.scheduler.optimizer.zero_grad()
            loss.backward()
            self.scheduler.optimizer.step()
            ls.append(loss.item())
        
        self.scheduler._add_metric(self.smiles_path)
        if epoch % self.save_epochs == 0:
            data = basename(self.smiles_path)
            self.prior.save(f'{self.save_path}.{data[:-4]}' if epoch == self.epochs - 1 else f'{self.save_path}.{data[:-4]}.{epoch}' )
        
        return np.mean(ls).item()





