from RL.reinforcement import Reinforcement
from configurations import REINFORCEMENT_CONFIG_RANDOM_AGENT_ITS_PRODUCT_EXPLORATION as Explo
from configurations import REINFORCEMENT_CONFIG_RANDOM_AGENT_MURCKO_PRODUCT_EXPLORE_AND_EXPLOIT as Murcko
from configurations import REINFORCEMENT_CONFIG_RANDOM_AGENT_ITS_PRODUCT_EXPLOITATION as Exploit
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
    if os.path.exists(os.path.join(csv_folder_path, f'likelihoods_scores_valid_smiles_{name}.csv')):
        
        df = pd.read_csv(os.path.join(csv_folder_path, f'likelihoods_scores_valid_smiles_{name}.csv'))

        
        fig, (l, s, vs) = plt.subplots(3, 1, sharex=True)
        fig.set_figwidth(15)
        fig.set_figheight(5)
        l.set_title('Mean Likelihoods per step')
        l.plot(df['Agent_likelihoods'], label='Agent\nLikelihood')
        l.plot(df['Prior_likelihoods'], label='Prior\nLikelihood')
        l.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        s.set_title('Mean Score per step')
        s.plot(df['Scores'])

        vs.set_title('Valid SMILES percentage generated per step')
        vs.plot(df['Valid_percentage'])

        if not os.path.exists('./figures'):
            os.makedirs('./figures')
        
        fig.savefig(f'./figures/plots_l_together_{name}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

        fig, ((al), (pl), (s), (vs)) = plt.subplots(4, 1, sharex=True)
        fig.set_figwidth(15)
        fig.set_figheight(7)

        al.set_title('Mean Agent Likelihoods per step')
        al.plot(df['Agent_likelihoods'], label='Agent\nLikelihood')

        
        pl.set_title('Mean Prior Likelihoods per step')
        pl.plot(df['Prior_likelihoods'], label='Prior\nLikelihood')

        s.set_title('Mean Score per step')
        s.plot(df['Scores'])

        vs.set_title('Valid SMILES percentage generated per step')
        vs.plot(df['Valid_percentage'])

        
        fig.savefig(f'./figures/plots_l_divided_{name}.png')
        plt.show(block=False)
        plt.pause(3)
        plt.close()

if __name__ == '__main__':
    
    log = rkl.logger()
    log.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    
    R_Exploit = Reinforcement(Exploit['batch_size'], Exploit['sigma'], Exploit['f'], 
                             Exploit['scoring_func'], Exploit['buffer'], Exploit['prior_path'],
                             Exploit['agent_path'], Exploit['lr'])
    
    R_Murcko = Reinforcement(Murcko['batch_size'], Murcko['sigma'], Murcko['f'], 
                             Murcko['scoring_func'], Murcko['buffer'], Murcko['prior_path'],
                             Murcko['agent_path'], Murcko['lr'])
    
    
    R_Explo = Reinforcement(Explo['batch_size'], Explo['sigma'], Explo['f'], 
                             Explo['scoring_func'], Explo['buffer'], Explo['prior_path'],
                             Explo['agent_path'], Explo['lr'])
    
    run(R_Exploit, 200, 'csvs', Exploit['name'])
    run(R_Murcko, 200, 'csvs', Murcko['name'])
    run(R_Explo, 200, 'csvs', Explo['name'])
    
    plots('csvs', Exploit['name'])
    plots('csvs', Murcko['name'])
    plots('csvs', Explo['name'])