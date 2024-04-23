from abc import abstractmethod
from ObservationModels.ObservationModel import ObservationModel
import numpy as np

class Obstacle(ObservationModel):
    @abstractmethod
    def checkCollision(self, state):
        pass
    
    @abstractmethod
    def getObservation(self, state, ref):
        pass
    
    @abstractmethod
    def getHMatrix(self, state, ref):
        return None