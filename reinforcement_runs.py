from RL.reinforcement import Reinforcement
from configurations import REINFORCEMENT_CONFIG_AURORA_AGENT_ITS_PRODUCT as Auro,\
      REINFORCEMENT_CONFIG_BRAF_AGENT_ITS_PRODUCT as Braf,\
        REINFORCEMENT_CONFIG_RANDOM_AGENT_ITS_PRODUCT_EXPLORATION as Explo
import os
import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import rdkit.RDLogger as rkl
import rdkit.rdBase as rkrb

def run(reinforcement: Reinforcement, steps: int, csv_folder_path: str, name: str):
    if not os.path.exists(csv_folder_path):
        os.makedirs(csv_folder_path)
    if not os.path.exists('./logs'):
        os.mkdir('./logs')
    with open(f'./logs/{name}.txt', 'a') as out:
        reinforcement.run(steps, out, csv_folder_path, name)

def plots(csv_folder_path: str, name: str):
    if os.path.exists(os.path.join(csv_folder_path, f'likelihoods_scores_valid_smiles_{name}')):
        
        df = pd.read_csv(os.path.join(csv_folder_path, f'likelihoods_scores_valid_smiles_{name}'))

        
        fig, (l, s, vs) = plt.subplots(3, 1, sharex=True)
        fig.set_figwidth(15)
        fig.set_figheight(5)
        l.set_title('Mean Likelihoods per step')
        l.plot(df['Agent_likelihoods'], label='Agent\nLikelihood')
        l.plot(df['Prior_likelihoods'], label='Prior\nLikelihood')
        l.legend(loc="upper left", shadow=True, fancybox=True)

        s.set_title('Mean Score per step')
        s.plot(df['Scores'])

        vs.set_title('Valid SMILES percentage generated per step')
        vs.plot(df['Valid_percentage'])

        if os.path.exists('./figures'):
            os.makedirs('./figures')
        
        fig.savefig(f'plots_l_together_{name}.png')

        fig, ((al), (pl), (s), (vs)) = plt.subplots(4, 1, sharex=True)
        fig.set_figwidth(15)
        fig.set_figheight(5)

        al.set_title('Mean Agent Likelihoods per step')
        al.plot(df['Agent_likelihoods'], label='Agent\nLikelihood')

        
        al.set_title('Mean Prior Likelihoods per step')
        pl.plot(df['Prior_likelihoods'], label='Prior\nLikelihood')

        s.set_title('Mean Score per step')
        s.plot(df['Scores'])

        vs.set_title('Valid SMILES percentage generated per step')
        vs.plot(df['Valid_percentage'])

        
        fig.savefig(f'plots_l_divided_{name}.png')

if __name__ == '__main__':
    
    log = rkl.logger()
    log.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    
    R_Aurora = Reinforcement(Auro['batch_size'], Auro['sigma'], Auro['f'], 
                             Auro['scoring_func'], Auro['buffer'], Auro['prior_path'],
                             Auro['agent_path'], Auro['lr'])

    R_Braf = Reinforcement(Braf['batch_size'], Braf['sigma'], Braf['f'], 
                             Braf['scoring_func'], Braf['buffer'], Braf['prior_path'],
                             Braf['agent_path'], Braf['lr'])
    
    
    R_Explo = Reinforcement(Explo['batch_size'], Explo['sigma'], Explo['f'], 
                             Explo['scoring_func'], Explo['buffer'], Explo['prior_path'],
                             Explo['agent_path'], Explo['lr'])
    
    run(R_Aurora, 500, 'csvs', Auro['name'])
    run(R_Braf, 500, 'csvs', Braf['name'])
    run(R_Explo, 500, 'csvs', Explo['name'])
    
    plots('csvs', Auro['name'])
    plots('csvs', Braf['name'])
    plots('csvs', Explo['name'])