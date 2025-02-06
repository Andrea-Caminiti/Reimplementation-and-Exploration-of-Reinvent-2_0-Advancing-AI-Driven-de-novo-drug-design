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
    '''
    Function to get canonical rdkit SMILES string
    Params:
    :param smile: (str) SMILES string to make canonical
    '''
    return rk.MolToSmiles(rk.MolFromSmiles(smile, sanitize=False), isomericSmiles=False)

#Abstract class
class ScaffoldFilter:
    '''
    Abstract Class, parent of any filter class
    '''
    def __init__(self, parameters: ScaffoldParameters):
        self.scaffolds = {}
        self.parameters = parameters

    def score(self, summary: ScoreSummary):
        '''
        Method to get the actual scores of the SMILES strings
        Params:
        :param summary: (ScoreSummary) score summary obtained from the scoring function
        '''
        pass

    def compute_scaffold(self, smile: str):
        '''
        Method to get the correct scaffold for the SMILES string
        Params:
        :param smile: (str) SMILES string to get the scaffold for
        '''
        pass

    def smiles_already_in_scaffold(self, scaffold: str, smile: str):
        '''
        Method to check if the SMILES string in input already is in the given scaffold
        Params:
        :param scaffold: (str) Scaffold to search into
        :param smile: (str) SMILES string to search for
        '''
        return self.scaffolds.get(scaffold, {}).get(smile, False)

    def add_to_scaffold(self, index: int, score: float, smile: str, scaffold: str, components: List[Component]): 
        '''
        Method to add SMILES string to given scaffold
        Params:
        :param index: (int) Index of the SMILES in the score summary
        :param score: (float) total_score from the score summary
        :param smile: (str) SMILES string to add to scaffold
        :param scaffold: (str) Scaffold to add to
        :param components: (List[Component]) List of the components used in the scoring function
        '''
        component_scores = {c.parameters.name: float(c.total_score[index]) for c in components}
        component_scores['total_score'] = score
        if scaffold in self.scaffolds:
            self.scaffolds[scaffold][smile] = component_scores
        else:
            self.scaffolds[scaffold] = {smile: component_scores}

    def penalize(self, scaffold: str, score: float):
        '''
        Method to penalize the score if the scaffold is full
        Params:
        :param scaffold: (str) scaffold to look into
        :param score: (float) score to return if the scaffold is not full
        '''
        return score if len(self.scaffolds.get(scaffold, {})) < self.parameters.nbmax else 0.0

class NoScaffoldFilter(ScaffoldFilter):
    '''
    Use no scaffold, each smile is its own scaffold, used for exploration
    '''
    def __init__(self, parameters):
        super().__init__(parameters)

    def score(self, summary: ScoreSummary):
        '''
        Method to get the actual scores of the SMILES strings
        Params:
        :param summary: (ScoreSummary) score summary obtained from the scoring function
        '''
        scores = summary.total_score
        smiles = summary.smiles

        for index in summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            
            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, smile, summary.components)
            
        return scores

class ScaffoldSimilarityFilter(ScaffoldFilter):
    '''
    All side chains are stripped.
    Murcko Scaffold Filter for 'minsimilarity' = 1.0
    '''
    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)
        self.fingerprints = {}

    def score(self, summary: ScoreSummary):
        '''
        Method to get the actual scores of the SMILES strings.
        It automatically computes the scaffold and penalizes it depending 
        if the scaffold is full or the SMILES string is already there
        Params:
        :param summary: (ScoreSummary) score summary obtained from the scoring function
        '''
        scores = summary.total_score
        smiles = summary.smiles

        for index in summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            scaffold = self.compute_scaffold(smile)
            scaffold = self.similar_to(scaffold)
            scores[index] = 0.0 if self.smiles_already_in_scaffold(scaffold, smile) else scores[index]

            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, scaffold, summary.components)
                scores[index] = self.penalize(scaffold, scores[index])
            
        return scores
        
    def compute_scaffold(self, smile: str):
        '''
        Method to compute the scaffold given a SMILES string
        Params:
        :param smile: (str) SMILES string to find the scaffold for
        '''
        mol = rk.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = Scaffolds.GetScaffoldForMol(mol)
                smiles = rk.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                smiles = ''
            return smiles
        return ''

    def similar_to(self, scaffold: str):
        '''
        Checks if a similar scaffold already exists and if so, uses that one
        '''
        if scaffold != '':
            fp = Pairs.GetAtomPairFingerprint(rk.MolFromSmiles(scaffold))
            fps = list(self.fingerprints.values())

            if len(fps) > 0:
                similarity_scores = DataStructs.BulkDiceSimilarity(fp, fps)
                closest = np.argmax(similarity_scores)
                if similarity_scores[closest] >= self.parameters.minsimilarity:
                    scaffold = list(self.fingerprints.keys())[closest]
                    fp = self.fingerprints[scaffold]

            self.fingerprints[scaffold] = fp
        return scaffold

class IdenticalTopologicalScaffold(ScaffoldFilter):
    '''Removes  all  side chains  and  converts all  atoms  in  the  remaining scaffold to sp3 carbons.'''

    def __init__(self, parameters: ScaffoldParameters):
        super().__init__(parameters)

    def score(self, summary: ScoreSummary):
        '''
        Method to get the actual scores of the SMILES strings.
        It automatically computes the scaffold and penalizes it depending 
        if the scaffold is full or the SMILES string is already there
        Params:
        :param summary: (ScoreSummary) score summary obtained from the scoring function
        '''
        scores = summary.total_score
        smiles = summary.smiles

        for index in summary.valid_indexes:
            smile = rk_smiles(smiles[index])
            scaffold = self.compute_scaffold(smile)
            scores[index] = 0 if self.smiles_already_in_scaffold(scaffold, smile) else scores[index]
            if scores[index] >= self.parameters.minscore:
                self.add_to_scaffold(index, scores[index], smile, scaffold, summary.components)
                scores[index] = self.penalize(scaffold, scores[index])
        return scores

    def compute_scaffold(self, smile: str):
        '''
        Method to compute the scaffold given a SMILES string
        Params:
        :param smile: (str) SMILES string to find the scaffold for
        '''
        mol = rk.MolFromSmiles(smile)
        if mol:
            try:
                scaffold = Scaffolds.MakeScaffoldGeneric(Scaffolds.GetScaffoldForMol(mol))
                smiles = rk.MolToSmiles(scaffold, isomericSmiles=False)
            except ValueError:
                smiles = ''
            return smiles
        return ''