import os
from typing import List

import numpy as np
import pandas as pd
import torch
import rdkit.Chem as rk
from rdkit import DataStructs
import rdkit.Chem.Scaffolds.MurckoScaffold as Scaffolds
from rdkit.Chem.AtomPairs import Pairs

from RL.parameters import ScaffoldParameters
from RL.summaries import ScoreSummary, Component


def rk_smiles(smile):
    return rk.MolToSmiles(rk.MolFromSmiles(smile, sanitize=False), isomericSmiles=False)
#Abstract class
class ScaffoldFilter:

    def __init__(self, parameters: ScaffoldParameters):
        self.scaffolds = {}
        self.parameters = parameters

    def score(self, summary: ScoreSummary):
        pass

    def compute_scaffold(self, smile: str):
        pass

    def smiles_already_in_scaffold(self, scaffold: str, smile: str):

        return self.scaffolds.get(scaffold, {}).get(smile, False)

    def add_to_scaffold(self, index: int, score: float, smile: str, scaffold: str, components: List[Component]): 
        component_scores = {c.parameters.name: float(c.total_score[index]) for c in components}
        component_scores['total_score'] = score
        if scaffold in self.scaffolds:
            self.scaffolds[scaffold][smile] = component_scores
        else:
            self.scaffolds[scaffold] = {smile: component_scores}

    def save_to_csv(self, save_path: str, run: int):
        if not os.path.isdir(save_path):
            os.makedirs(save_path)

        df = {'Scaffold': [], 'SMILES': []}

        for scaffold in self.scaffolds.keys():
            for smile, score in self.scaffolds[scaffold].items():
                df['Scaffold'].append(scaffold)
                df['SMILES'].append(smile)
                for component, value in score:
                    df[component] = df.get(component, []).append(value)

        full_path = os.path.join(save_path, f'scaffolds_run_{run}.csv')
        df = pd.DataFrame(df)
        if len(df) > 0:
            df = df.sort_values(by='total_score', ascending=False)
            df.to_csv(full_path)

    def penalize(self, scaffold: str, score: float):
        return score if len(self.scaffolds.get(scaffold, {})) < self.parameters.nbmax else 0.0

class NoScaffoldFilter(ScaffoldFilter):
    '''
    Each smile is its own scaffold
    '''
    def __init__(self, parameters):
        super().__init__(parameters)

    def score(self, summary: ScoreSummary):
        scores = summary.total_score
        smiles = summary.smiles

        for index in summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            
            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, smile, summary.components)
            
            return scores

class ScaffoldSimilarityFilter(ScaffoldFilter):
    '''
    Murcko Scaffold Filter for 'minsimilarity' = 1.0
    '''
    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)
        self.fingerprints = {}

    def score(self, summary: ScoreSummary):
        scores = summary.total_score
        smiles = summary.smiles

        for index in summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            scaffold = self.compute_scaffold(smile)
            scaffold = self.similar_to(scaffold)
            scores[index] = 0.0 if self.smiles_already_in_scaffold(smile) else scores[index]

            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, scaffold, summary.components)
                scores[index] = self.penalize(scaffold, scores[index])
            
            return scores
        
    def compute_scaffold(self, smile):
        mol = rk.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = Scaffolds.GetScaffoldForMol(mol)
                smiles = rk.MolToSmiles(mol, isomericSmiles=False)
            except ValueError:
                smiles = ''
            return smiles
        return ''

    def similar_to(self, scaffold):
        
        if scaffold != '':
            fp = Pairs.GetAtomPairFingerprint(rk.MolFromSmiles(scaffold))

            # make a list of the stored fingerprints for similarity calculations
            fps = list(self.fingerprints.values())

            # check, if a similar scaffold entry already exists and if so, use this one instead
            if len(fps) > 0:
                similarity_scores = DataStructs.BulkDiceSimilarity(fp, fps)
                closest = np.argmax(similarity_scores)
                if similarity_scores[closest] >= self.parameters.minsimilarity:
                    scaffold = list(self.fingerprints.keys())[closest]
                    fp = self.fingerprints[scaffold]

            self.fingerprints[scaffold] = fp
        return scaffold

class IdenticalTopologicalScaffold(ScaffoldFilter):
    """Penalizes compounds based on exact Topological Scaffolds previously generated."""

    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)

    def score(self, score_summary: ScoreSummary):
        scores = score_summary.total_score
        smiles = score_summary.smiles

        for index in score_summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            scaffold = self._calculate_scaffold(smile)
            scores[index] = 0 if self.smiles_already_in_scaffold(scaffold, smile) else scores[index]
            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, scaffold, score_summary.scaffold_log)
                scores[index] = self.penalize(scaffold, scores[index])
        return scores

    def _calculate_scaffold(self, smile: str):
        mol = rk.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = Scaffolds.MakeScaffoldGeneric(Scaffolds.GetScaffoldForMol(mol))
                smiles = rk.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                smiles = ''
            return smiles
        return ''