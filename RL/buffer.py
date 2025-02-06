import pandas as pd
import numpy as np
from typing import List
import rdkit.Chem as rk
from torch import FloatTensor

from RL.scoring_functions import ScoringFunction
from Prior.model import Prior

class Buffer():
    '''
    Buffer to store the best scoring compounds and give an head start for exploitation
    '''
    def __init__(self, size: int, smiles: List[str], sample_size: int,
                 scoring_func: ScoringFunction, prior: Prior):
        '''
        Params:
        :param size: (int) Size of the buffer
        :param smiles: (List[str]) SMILES to initialize the buffer with, can be empty
        :param sample_size: (int) Amount of SMILES to sample from memory to pass to the agent
        :param scoring_func: (ScoringFunction) Scoring function to score the input SMILES
        :param prior: (Prior) Prior to compute the likelihood of the input SMILES
        '''
        self.size = size
        self.smiles = smiles
        self.sample_size = sample_size
        self.scoring_func = scoring_func
        self.prior = prior

        self.memory = pd.DataFrame(columns=['smiles', 'score', 'likelihood'])
        if len(self.smiles) > 0:
            rk_smiles = [rk.MolToSmiles(rk.MolFromSmiles(smile, sanitize=False), isomericSmiles=False) for smile in self.smiles]
            rk_smiles = [smile for smile in rk_smiles if smile != None]
            self.eval_and_add(rk_smiles, self.scoring_func, self.prior)
        
    def eval_and_add(self, smiles: List[str], scoring_func: ScoringFunction, prior: Prior):
        '''
        Function to score and add the input SMILES to memory
        Params:
        :param smiles: (List[str]) SMILES to score and add to memory
        :param scoring_func: (ScoringFunction) Scoring function to score the input SMILES
        :param prior: (Prior) Prior to compute the likelihood of the input SMILES
        '''
        if len(smiles) > 0:
            score = scoring_func.final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            df = pd.DataFrame({"smiles": smiles, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory._append(df)
            self.purge()

    def purge(self):
        '''
        Method to clean the memory from duplicates and invalid SMILES
        '''
        df = self.memory.drop_duplicates(subset=['smiles'])
        df.reset_index(drop=True, inplace=True)
        mols = [rk.MolFromSmiles(smile) for smile in df['smiles']]
        valid = np.array([0 if mol is None else 1 for mol in mols])
        invalid_indexes = np.where(valid == 0)
        df.drop(invalid_indexes[0])
        df = df.sort_values(by='score', ascending=False)
        df.reset_index(drop=True, inplace=True)
        self.memory = df.iloc[:self.size]
    
    def add(self, smiles: List[str], score: List[float], neg_likelihood: List[float]):
        '''
        Method to add SMILES having it already scored
        Params:
        :param smiles: (List[str]) SMILES to add to memory
        :param score: (List[float]) Scores for the input SMILES string
        :param neg_likelihood: (List[float]) Negative Log Likelihood for the input SMILES
        '''
        df = pd.DataFrame({"smiles": smiles, "score": score.detach().cpu().numpy(), "likelihood": neg_likelihood.detach().cpu().numpy()})
        self.memory = self.memory._append(df)
        self.purge()

    def sample(self):
        '''
        Method to sample from memory
        '''
        sample_size = min(len(self.memory), self.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size, replace=False)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], []

    def log_out_memory(self, path: str):
        '''
        Method to save memory to file
        Params:
        :param path: (str) File path to save to
        '''
        self.memory.to_csv(path)