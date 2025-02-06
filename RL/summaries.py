import numpy as np
from typing import List

from RL.parameters import ComponentParameters

class Component:
    '''
    Class to store the summary for each component
    '''
    def __init__(self, score: np.array, parameters: ComponentParameters):
        self.total_score = score
        self.parameters = parameters

class ScoreSummary():
    '''
    Class to store the summary for the score of the SMILES strings
    '''
    def __init__(self, total_score: np.array, smiles: List[str], 
                 valid_indexes: List[int], components: List[Component]):
        self.total_score = total_score
        self.smiles = smiles
        self.valid_indexes = valid_indexes
        self.components = components