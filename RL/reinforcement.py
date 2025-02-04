import torch
import numpy as np
from io import TextIOWrapper
from typing import List
import os
import pandas as pd
import rdkit.Chem as rk
from tqdm import tqdm

from Prior.model import Prior
from RL.filters import ScaffoldFilter
from RL.scoring_functions import ScoringFunction
from RL.buffer import Buffer

class Reinforcement:
    def __init__(self,
                 batch_size: int, sigma: int,
                 filter: ScaffoldFilter, scoring_func: ScoringFunction,
                 buffer: Buffer,
                 prior_path: str, 
                 agent_path: str,
                 lr: float = 1e-5
                 ):

        self.prior = Prior().load_prior(prior_path)
        self.agent = Prior().load_prior(agent_path)
        
        self.agent.Rnn.to('cuda')
        self.prior.Rnn.to('cuda')

        self.agent_path = agent_path
        self.prior_path = prior_path

        assert self.prior.vocabulary == self.agent.vocabulary
        
        self.freeze_prior()

        self.buffer = buffer

        self.filter = filter
        self.scoring_func = scoring_func

        self.stats = []
        self.optimizer = torch.optim.Adam(self.agent.Rnn.parameters(), lr=lr)

        self.batch_size = batch_size
        self.sigma = sigma

    def freeze_prior(self):
        for param in self.prior.Rnn.parameters():
            param.requires_grad = False
    
    def run(self, steps: int, out_file: TextIOWrapper, csv_folder_path: str, name: str):
        out_file.write(f'Starting a {steps} steps run\n')
        self.freeze_prior()
        a_likelihoods = []
        p_likelihoods = []
        scores = []
        valid_smiles = []
        losses = []
        for step in tqdm(range(steps), desc=f'Run', leave=False):
            sequences, smiles, a_likelihood = self.agent.sample_sequences_and_smiles(self.batch_size)
            _  , indexes = np.unique(smiles, return_index=True)
            indexes = np.sort(indexes)
            sequences = sequences[indexes]
            smiles = np.array(smiles)[indexes]
            a_likelihood = -a_likelihood[indexes]
            p_likelihood = -self.prior.likelihood(sequences)
            summary = self.scoring_func.final_score(smiles)
            score = self.filter.score(summary)
            score = torch.from_numpy(score)
            score = torch.autograd.Variable(score)
            if torch.cuda.is_available():
                score = score.cuda()
            loss = self.augment_loss(p_likelihood, a_likelihood, score)
            loss, a_likelihood = self.buffer_filter(self.agent, loss, a_likelihood, p_likelihood, smiles, score)

            loss = loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            mols = [rk.MolFromSmiles(smile) for smile in smiles]
            valid = [0 if mol is None else 1 for mol in mols]
            valid_smiles.append(np.sum(valid)/len(smiles))
            a_likelihoods.append(a_likelihood.mean().cpu().detach().numpy())
            p_likelihoods.append(p_likelihood.mean().cpu().detach().numpy())
            scores.append(score.mean().cpu().detach().numpy())
            losses.append(loss.detach().cpu().numpy())
        
        out_file.write('End of the run\nSummary:\n')
        out_file.write(f'Average Agent Likelihood: {np.mean(a_likelihoods)}\n')
        out_file.write(f'Average Prior Likelihood: {np.mean(p_likelihoods)}\n')
        out_file.write(f'Average Augmented Loss: {np.mean(losses)}\n')
        out_file.write(f'Average Score: {np.mean(scores)}\n')
        out_file.write(f'Average Valid Smiles: {np.mean(valid_smiles)}\n\n\n')
        self.buffer.log_out_memory(os.path.join(csv_folder_path, f'buffer_memory_{name}.csv'))
        df = pd.DataFrame({'Agent_likelihoods': a_likelihoods, 
                           'Prior_likelihoods': p_likelihoods, 
                           'Scores': scores, 
                           'Valid_percentage': valid_smiles })
        df.to_csv(os.path.join(csv_folder_path, f'likelihoods_scores_valid_smiles_{name}.csv'))


    def augment_loss(self, prior_likelihood: torch.FloatTensor, agent_likelihood: torch.FloatTensor, score: torch.autograd.Variable):
            if type(prior_likelihood) != torch.Tensor:
                prior_likelihood = torch.from_numpy(prior_likelihood)
            if type(agent_likelihood) != torch.Tensor:
                agent_likelihood = torch.from_numpy(agent_likelihood)
            if type(score) != torch.Tensor:
                score = torch.from_numpy(score)

            augmented = prior_likelihood + self.sigma * score
            return torch.pow((augmented.to('cuda') - agent_likelihood.to('cuda')), 2)
    
    def buffer_filter(self, agent: Prior, loss: torch.FloatTensor,
                      agent_likelihood: torch.FloatTensor, 
                      prior_likelihood: torch.FloatTensor,
                      smiles: List[str], score: torch.FloatTensor):
        if self.buffer != None:
            exp_smiles, exp_scores, exp_prior_likelihood = self.buffer.sample()
            if len(exp_smiles) > 0:
                exp_agent_likelihood = -agent.likelihood_smiles(exp_smiles)
                exp_loss = self.augment_loss(exp_prior_likelihood, exp_agent_likelihood, exp_scores)
                loss = torch.cat((loss, exp_loss), 0)
                agent_likelihood = torch.cat((agent_likelihood, exp_agent_likelihood), 0)
            self.buffer.add(smiles, score, prior_likelihood)
        return loss, agent_likelihood