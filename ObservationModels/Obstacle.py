from abc import abstractmethod
from copy import copy
from ObservationModels.ObservationModel import ObservationModel
import numpy as np
from types import MethodType


class Obstacle(ObservationModel):
    @abstractmethod
    def checkCollision(self, state):
        pass
    
    def __add__(self, b):
        temp_obstacle = self.__add__(self, b)
        
        temp_obstacle.checkCollision = MethodType(lambda self, state, ref: self.checkCollision(state) or b.checkCollision(state), self)
        
        return temp_obstacle