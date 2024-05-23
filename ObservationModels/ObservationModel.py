from abc import ABC, abstractmethod
from copy import copy
import numpy as np
from types import MethodType

class ObservationModel(ABC):
    @abstractmethod
    def getObservation(self, state, ref):
        pass
    
    @abstractmethod
    def getHMatrix(self, state, ref):
        pass
    
    def __add__(self, b):
        temp_observation = copy(self)
        
        temp_observation.getObservation = MethodType(lambda self, state, ref: np.hstack([self.getObservation(state, ref), b.getObservation(state, ref)]), self)
        temp_observation.getHMatrix = MethodType(lambda self, state, ref: np.vstack([self.getHMatrix(state, ref), b.getHMatrix(state, ref)]), self)
        
        return temp_observation