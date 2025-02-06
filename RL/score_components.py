from typing import List, Dict
from rdkit import Chem as rk
from rdkit.Chem.Lipinski import NumHDonors,NumRotatableBonds
from rdkit.Chem.Descriptors import MolWt,TPSA
from rdkit.Chem.rdMolDescriptors import CalcNumRings
import numpy as np
import math

from RL.parameters import ComponentParameters
from RL.summaries import Component

class ScoreComponent():
    '''
    Parent Class of all components
    '''
    def __init__(self, parameters: ComponentParameters):
        self.parameters = parameters

    def score(self, mols: List):
        pass

    def smiles_to_fingerprints(self, smiles: List[str], radius=3, useCounts=True, useFeatures=True):
        mols, idx = self.smiles_to_mols(smiles)
        fingerprints = [rk.AllChem.GetMorganFingerprint(
            mol,
            radius,
            useCounts=useCounts,
            useFeatures=useFeatures
        ) for mol in mols]
        return fingerprints, idx
    
    def smiles_to_mols(self, query_smiles: List[str]):
        mols = [rk.MolFromSmiles(smile) for smile in query_smiles]
        valid = [0 if mol is None else 1 for mol in mols]
        valid_idxs = [idx for idx, boolean in enumerate(valid) if boolean == 1]
        valid_mols = [mols[idx] for idx in valid_idxs]
        return valid_mols, valid_idxs
    
class CustomAlerts(ScoreComponent):
    '''
    Penalty Component, requires a set of SMARTS strings to look for in the generated SMILES.
    If found returns 0.
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        if len(self.parameters.smarts) > 0:
            self.custom_alerts = self.parameters.smarts
        else: 
            self.custom_alerts = ['']
    
    def score(self, mols: List):
        score = self.match(mols, self.custom_alerts)
        score_summary = Component(score=score, parameters=self.parameters)
        return score_summary
    
    def match(self, mols: List, smarts: List[str]):
        m = np.array([any([mol.HasSubstructMatch(rk.MolFromSmarts(s)) for s in smarts
                      if rk.MolFromSmarts(s)]) for mol in mols], dtype=np.int32)
        
        return 1 - m
    
class JaccardDistance(ScoreComponent):
    '''
    Non penalty component. Requires a set of SMILES strings to return
    the lowest distance score to.
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters) 
        self.fingerprints, self.indexes = self.smiles_to_fingerprints(self.parameters.smiles)
       
    def score(self, mols: List):
        fp = [rk.AllChem.GetMorganFingerprint(
            mol,
            radius = 3,
            useCounts=True,
            useFeatures=True
        ) for mol in mols]
        
        score = self.jd(self, fp, self.fingerprints)

        return Component(score, self.parameters)
    
    def jd(self, queries: List, references: List):
        score = []
        for query in queries:
            all_distances = [1 - rk.DataStructs.TanimotoSimilarity(query, reference) for reference in references]
            closest = min(all_distances)
            score.append(closest)
        return np.array(score, dtype=np.int32)
    
class MatchingSubstructure(ScoreComponent):
    '''
    Penalty component. Requires a set of SMARTS strings to check for 
    similar substructure. Returns 1 if found. 
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def score(self, mols: List):
        score = self.match(mols, self.parameters.smarts)
        return Component(total_score=score, parameters=self.parameters)

    def match(self, mols: List, smarts: List[str]):
        if len(smarts) == 0:
            return np.ones(len(mols), dtype=np.float32)

        m = np.array([any([mol.HasSubstructMatch(rk.MolFromSmarts(s)) for s in smarts
                      if rk.MolFromSmarts(s)]) for mol in mols], dtype=np.int32)
        
        return 0.5 * (1 + np.array(m))

class QEDScore(ScoreComponent):
    '''
    Non penalty component.
    Computes the quantitative estimation of drug-likeness score
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
    
    def score(self, mols: List):

        qeds = []
        for mol in mols:
            try:
                qed_score = rk.Descriptors.qed(mol)
            except ValueError:
                qed_score = 0.0
            qeds.append(qed_score)
        
        score = np.array(qeds, dtype=np.float32)
        return Component(score, self.parameters)
    
class TanimotoSimilarity(ScoreComponent):
    ''' 
    Non penalty component.
    Requires a set of SMILES and returns the highest similarity score to it'''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.fingerprints, self.indexes = self.smiles_to_fingerprints(self.parameters.smiles)

    def score(self, mols: List):
        fps = self.mols_to_fingerprints(mols)
        score = np.array([np.max(rk.DataStructs.BulkTanimotoSimilarity(fp, self.fingerprints)) for fp in fps])
    
        return Component(score, self.parameters)
        

    def mols_to_fingerprints(self, mols: List):
        fps = [rk.AllChem.GetMorganFingerprint(
            mol,
            radius=3,
            useCounts=True,
            useFeatures=True
        ) for mol in mols]
        return fps

class TransformationFactory():
    def __init__(self):
        self._transformation_function_registry = self.default_registry()

    def default_registry(self) -> dict:
        
        transformation_list = {
            'sigmoid': self.sigmoid_transformation,
            'reverse_sigmoid': self.reverse_sigmoid_transformation,
            'double_sigmoid': self.double_sigmoid,
            'no_transformation': self.no_transformation,
            'step': self.step
        }
        return transformation_list

    def get_transformation_function(self, parameters: dict):
        transformation_type = parameters['transformation_type']
        transformation_function = self._transformation_function_registry[transformation_type]
        return transformation_function

    def no_transformation(self, score: list):
        return np.array(score, dtype=np.float32)

    def step(self, score, parameters):
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]

        def formula(value, low, high):
            if low <= value <= high:
                return 1
            return 0

        transformed = [formula(value, _low, _high) for value in score]
        return np.array(transformed, dtype=np.float32)

    def sigmoid_transformation(self, score: list, parameters: dict):
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _k = parameters[self._csp_enum.K]

        def exp(value, low, high, k) -> float:
            return math.pow(10, (10 * k * (value - (low + high) * 0.5) / (low - high)))

        transformed = [1 / (1 + exp(value, _low, _high, _k)) for value in score]
        return np.array(transformed, dtype=np.float32)

    def reverse_sigmoid_transformation(self, score: list, parameters: dict) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _k = parameters[self._csp_enum.K]

        def formula(value, low, high, k) -> float:
            try:
                return 1 / (1 + 10 ** (k * (value - (high + low) / 2) * 10 / (high - low)))
            except:
                return 0

        transformed = [formula(value, _low, _high, _k) for value in score]
        return np.array(transformed, dtype=np.float32)

    def double_sigmoid(self, score: list, parameters: dict) -> np.array:
        _low = parameters[self._csp_enum.LOW]
        _high = parameters[self._csp_enum.HIGH]
        _coef_div = parameters[self._csp_enum.COEF_DIV]
        _coef_si = parameters[self._csp_enum.COEF_SI]
        _coef_se = parameters[self._csp_enum.COEF_SE]

        def formula(value, low, high, coef_div=100., coef_si=150., coef_se=150.):
            try:
                A = 10 ** (coef_se * (value / coef_div))
                B = (10 ** (coef_se * (value / coef_div)) + 10 ** (coef_se * (low / coef_div)))
                C = (10 ** (coef_si * (value / coef_div)) / (
                        10 ** (coef_si * (value / coef_div)) + 10 ** (coef_si * (high / coef_div))))
                return (A / B) - C
            except:
                return 0

        transformed = [formula(value, _low, _high, _coef_div, _coef_si, _coef_se) for value in score]
        return np.array(transformed, dtype=np.float32)

class PhysChemComponent(ScoreComponent):
    '''
    Base class for the Physical-Chimical component
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)
        self.transformation_function = self.transformation(self.parameters.parameters)

    def score(self, mols: List):
        scores = []
        for mol in mols:
            try:
                score = self.property(mol)
            except ValueError:
                score = 0.0
            scores.append(score)
        scores = self.transformation_function(scores, self.parameters.parameters)
        final_score = np.array(scores, dtype=np.float32)
        return Component(total_score=final_score, parameters=self.parameters)

    def property(self, mol):
       pass

    def transformation(self, specific_parameters: Dict):
        factory = TransformationFactory()
        if self.parameters.parameters['transformation']:
            transform_function = factory.get_transformation_function(specific_parameters)
        else:
            self.parameters.specific_parameters['transformation_type'] = 'no_transformation'
            transform_function = factory.no_transformation
        return transform_function

class HBD(PhysChemComponent):
    '''
    Non penalty component. Computes the number of hydrogen bond donors
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def property(self, mol):
        return NumHDonors(mol)
    
class MolWeight(PhysChemComponent):
    ''' 
    Non penalty component.
    Computes molecular weight'''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def property(self, mol):
        return MolWt(mol)

class NumRings(PhysChemComponent):
    '''
    Non penalty component.
    Computes the number of rings in the molecule
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def property(self, mol):
        return CalcNumRings(mol)

class RotatableBonds(PhysChemComponent):
    ''' 
    Non penalty component.
    Computes the number of rotatable bonds
    '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def property(self, mol):
        return NumRotatableBonds(mol)
    
class PSA(PhysChemComponent):
    '''
    Non penalty component. 
    Computes molecule polarity '''
    def __init__(self, parameters: ComponentParameters):
        super().__init__(parameters)

    def property(self, mol):
        return TPSA(mol)