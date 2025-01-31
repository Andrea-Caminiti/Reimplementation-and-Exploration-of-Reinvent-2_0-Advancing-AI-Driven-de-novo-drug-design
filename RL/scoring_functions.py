from typing import List
import rdkit.Chem as rk
import numpy as np
import math

from RL.parameters import ComponentParameters
from RL.summaries import ScoreSummary, Component
from RL.score_components import TanimotoSimilarity, JaccardDistance, \
                                 CustomAlerts, QEDScore, \
                                 MatchingSubstructure, \
                                 MolWeight, PSA, \
                                 RotatableBonds, HBD_Lipinski, \
                                 NumRings

COMPONENTS = {
            'matching_substructure': MatchingSubstructure,
            'tanimoto_similarity': TanimotoSimilarity, 
            'jaccard_distance': JaccardDistance, 
            'custom_alerts': CustomAlerts, 
            'qed_score': QEDScore, 
            'molecular_weight': MolWeight,
            'tpsa': PSA,
            'num_rotatable_bonds': RotatableBonds,
            'num_hbd_lipinski': HBD_Lipinski,
            'num_rings': NumRings,
        }

def total_score(summary: Component, query_length: int, valid_indices: List[int]):
    total_score = np.zeros(query_length, dtype=np.float32)
    assert len(valid_indices) == len(summary.total_score)
    for idx, value in zip(valid_indices, summary.total_score):
        total_score[idx] = value
    summary.total_score = total_score
    return summary

#Abstract class
class ScoringFunction():
    def __init__(self, parameters: List[ComponentParameters]):

        self.parameters = parameters
        self.components = [COMPONENTS.get(p.component_type)(p) for p in self.parameters]

    def non_penalty(self, components: List[Component], smiles: List[str]):
        pass

    def is_penalty(self, summary: Component):
        return (summary.parameters.component_type == 'matching_substructure') or (
                summary.parameters.component_type == 'custom_alerts')
    
    def penalty(self, components: List[Component], smiles: List[str]):
        penalty = np.ones(len(smiles), dtype=np.float32)

        for component in components:
            if self.is_penalty(component):
                penalty *= component.total_score
        
        return penalty
    
    def final_score(self, smiles: List[str]):
        
        mols = [rk.MolFromSmiles(smile) for smile in smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        indexes = [index for index, boolean in enumerate(valid) if boolean]
        mols = [mols[index] for index in indexes]
        length = len(smiles)
        summaries = [total_score(c.score(mols), length, indexes) for c in self.components]
        penalty = self.penalty(summaries, smiles)
        non_penalty = self.non_penalty(summaries, smiles)
        final_score = penalty * non_penalty

        return ScoreSummary(np.array(final_score, dtype=np.float32), smiles, indexes)
    
class CustomProduct(ScoringFunction):

    def __init__(self, parameters: List[ComponentParameters]):
        super().__init__(parameters)

    def pow(self, values, weight):
        y = [math.pow(value, weight) for value in values]
        return np.array(y, dtype=np.float32)

    def all_weights(self, summaries: List[Component]) -> int:
        all_weights = []

        for summary in summaries:
            if not self.is_penalty(summary):
                all_weights.append(summary.parameters.weight)
        return sum(all_weights)

    def non_penalty(self, summaries: List[Component], smiles: List[str]):
        product = np.full(len(smiles), 1, dtype=np.float32)
        all_weights = self.all_weights(summaries)

        for summary in summaries:
            if not self.is_penalty(summary):
                comp_pow = self.pow(summary.total_score, summary.parameters.weight / all_weights)
                product *= comp_pow

        return product

class CustomSum(ScoringFunction):

    def __init__(self, parameters: List[ComponentParameters]):
        super().__init__(parameters)

    def non_penalty(self, summaries: List[Component], smiles: List[str]):
        total_sum = np.zeros(len(smiles), dtype=np.float32)
        all_weights = 0.0

        for summary in summaries:
            if not self._component_is_penalty(summary):
                total_sum = total_sum + summary.total_score * summary.parameters.weight
                all_weights += summary.parameters.weight

        if all_weights == 0:
            return np.ones(len(smiles), dtype=np.float32)

        return total_sum / all_weights
