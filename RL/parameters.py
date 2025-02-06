from typing import List

class ScaffoldParameters():
    '''
    Class to collect scaffold parameters
    '''
    def __init__(self, name: str, minscore: float, nbmax: int, minsimilarity: float):
        self.name = name
        self.minscore = minscore
        self.nbmax = nbmax # Max number of compounds per bucket
        self.minsimilarity = minsimilarity


class ComponentParameters():
    '''
    Class to collect component parameters
    '''
    def __init__(self, component_type: str, name: str, weight: float,
                smiles: List[str], smarts: List[str], model_path: str, 
                specific_parameters: dict = None):
        
        self.component_type = component_type
        self.name = name
        self.weight = weight
        self.smiles = smiles
        self.smarts = smarts
        self.model_path = model_path
        self.parameters = specific_parameters