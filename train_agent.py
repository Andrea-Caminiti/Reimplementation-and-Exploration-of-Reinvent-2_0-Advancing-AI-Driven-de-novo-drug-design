
from Prior.trainer import LearningRate, Trainer
from Prior.model import Prior
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH_AURORA as Auro
from configurations import TRAIN_AGENT_CONFIG_FROM_SCRATCH_BRAF as Braf

if __name__ == '__main__':
    agent = Prior(use_cuda=True)

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
                      patience=Auro['patience_train'],
                      save_path=Auro['save_path'],
                      save_epochs=Auro['save_epochs'],
                      smiles_path=Auro['smiles_path'],
                      prior=agent, 
                      prior_path=Auro['prior_path'],
                      starting_epoch=Auro['starting_epoch']) 
    
    trainer.train()