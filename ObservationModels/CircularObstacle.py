from ObservationModels.Obstacle import Obstacle
import numpy as np

class CircularObstacle(Obstacle):
    def __init__(self, centre_pt, radius):
        self.centre_pt = centre_pt
        self.radius = radius
        
    def checkCollision(self, state):
        if (np.linalg.norm(self.centre_pt-state[0:2]) <= self.radius):
            return True
        
        return False
    def getObservation(self, state, ref):
        return (np.linalg.norm(self.centre_pt-state[0:2]) - self.radius)

    def getHMatrix(self, state, ref):
        return None