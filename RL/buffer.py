import pandas as pd
from typing import List
import rdkit.Chem as rk
from torch import FloatTensor

from RL.scoring_functions import ScoringFunction
from Prior.model import Prior

class Buffer():
    def __init__(self, size: int, smiles: List[str], sample_size: int,
                 scoring_func: ScoringFunction, prior: Prior):
        
        self.size = size
        self.smiles = smiles
        self.sample_size = sample_size
        self.scoring_func = scoring_func
        self.prior = prior

        self.memory = pd.DataFrame(columns=['smile', 'score', 'likelihood'])
        if len(self.smiles) > 0:
            rk_smiles = [rk.MolToSmiles(rk.MolFromSmiles(smile, sanitize=False), isomericSmiles=False) for smile in self.smiles]
            rk_smiles = [smile for smile in rk_smiles if smile != None]
            self.eval_and_add(rk_smiles, self.scoring_func, self.prior)
        
    def eval_and_add(self, smiles: List[str], scoring_func: ScoringFunction, prior: Prior):
        if len(smiles) > 0:
            score = scoring_func.final_score(smiles)
            likelihood = prior.likelihood_smiles(smiles)
            df = pd.DataFrame({"smiles": smiles, "score": score.total_score, "likelihood": -likelihood.detach().cpu().numpy()})
            self.memory = self.memory.append(df)
            self.purge()

    def purge(self):
        df = self.memory.drop_duplicates(subset=['smiles'])
        df.sort_values(by='score', inplace=True, ascending=bool)
        self.memory = df.iloc[:self.size]
    
    def add(self, smiles: List[str], score: List[float], neg_likelihood: FloatTensor):
        # NOTE: likelihood should be already negative
        df = pd.DataFrame({"smiles": smiles, "score": score, "likelihood": neg_likelihood.detach().cpu().numpy()})
        self.memory = self.memory.append(df)
        self.purge()

    def sample(self):
        sample_size = min(len(self.memory), self.sample_size)
        if sample_size > 0:
            sampled = self.memory.sample(sample_size, replace=False)
            smiles = sampled["smiles"].values
            scores = sampled["score"].values
            prior_likelihood = sampled["likelihood"].values
            return smiles, scores, prior_likelihood
        return [], [], []

    def log_out_memory(self, path: str):
        self.memory.to_csv(path)