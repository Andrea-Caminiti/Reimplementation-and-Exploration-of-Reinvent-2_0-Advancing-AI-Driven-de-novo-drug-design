TRAIN_AGENT_CONFIG_FROM_SCRATCH_AURORA = {
    #Learning rate config
    'prior_path': None,
    'mode': 'adaptive',
    'max_v': 5e-3, 
    'min_v': 1e-5, 
    'step':  1, 
    'decay': 0.8, 
    'sample_size': 128,
    'patience_lr': 5, 
    'validation': False,
    #Trainer config
    'epochs': 100,
    'bs': 128, 
    'early_stop': True,
    'patience_train': 5,
    'save_path': 'Agents/Agent',
    'save_epochs': 3,
    'smiles_path': 'data\Aurora-A_dataset.smi',
    'starting_epoch': 1
}

TRAIN_AGENT_CONFIG_FROM_SCRATCH_BRAF = {
    #Learning rate config
    'prior_path': None,
    'mode': 'adaptive',
    'max_v': 5e-3, 
    'min_v': 1e-5, 
    'step':  1, 
    'decay': 0.8, 
    'sample_size': 128,
    'patience_lr': 5, 
    'validation': False,
    #Trainer config
    'epochs': 100,
    'bs': 128, 
    'patience_train': 5,
    'save_path': 'Agents/Agent',
    'save_epochs': 3,
    'smiles_path': 'data\B-raf_dataset.smi',
    'starting_epoch': 1
}

