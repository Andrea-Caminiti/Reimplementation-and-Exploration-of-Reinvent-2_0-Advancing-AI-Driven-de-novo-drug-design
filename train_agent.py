
from Prior.trainer import LearningRate, Trainer
from Prior.model import Prior
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH_AURORA as Auro
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH_BRAF as Braf
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH as general
import torch
if __name__ == '__main__':
    agent = Prior(use_cuda=True).load_prior('priors/prior.dataset')

    lr = LearningRate(prior=agent, 
                      prior_path=Auro['prior_path'],
                      mode=Auro['mode'],
                      max_v=Auro['max_v'], 
                      min_v=Auro['min_v'], 
                      step=Auro['step'], 
                      decay=Auro['decay'], 
                      sample_size=Auro['sample_size'],
                      patience=Auro['patience_lr'], 
                      validation=Auro['validation'])
    
    trainer = Trainer(epochs=Auro['epochs'],
                      lr=lr,
                      bs=Auro['bs'],
                      early_stop=Auro['early_stop'],
                      patience=Auro['patience_train'],
                      save_path=Auro['save_path'],
                      save_epochs=Auro['save_epochs'],
                      smiles_path=Auro['smiles_path'],
                      prior=agent, 
                      prior_path=Auro['prior_path'],
                      starting_epoch=Auro['starting_epoch']
                      ) 
    
    trainer.train()

    
    agent = Prior(use_cuda=True).load_prior('priors/prior.dataset')

    lr = LearningRate(prior=agent, 
                      prior_path=Braf['prior_path'],
                      mode=Braf['mode'],
                      max_v=Braf['max_v'], 
                      min_v=Braf['min_v'], 
                      step=Braf['step'], 
                      decay=Braf['decay'], 
                      sample_size=Braf['sample_size'],
                      patience=Braf['patience_lr'], 
                      validation=Braf['validation'])
    
    trainer = Trainer(epochs=Braf['epochs'],
                      lr=lr,
                      bs=Braf['bs'],
                      early_stop=Braf['early_stop'],
                      patience=Braf['patience_train'],
                      save_path=Braf['save_path'],
                      save_epochs=Braf['save_epochs'],
                      smiles_path=Braf['smiles_path'],
                      prior=agent, 
                      prior_path=Braf['prior_path'],
                      starting_epoch=Braf['starting_epoch']
                      ) 
    
    trainer.train()