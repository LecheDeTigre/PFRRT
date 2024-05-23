from ObservationModels.Obstacle import Obstacle
import numpy as np

class CircularObstacle(Obstacle):
    def __init__(self, centre_pt, radius, weight):
        self.centre_pt = centre_pt
        self.radius = radius
        self.weight = weight
        
    def checkCollision(self, state):
        if (np.linalg.norm(self.centre_pt-state[0:2]) <= self.radius):
            return True
        
        return False
    def getObservation(self, state, ref):
        return np.exp(-0.5*self.weight*(np.linalg.norm(self.centre_pt-state[0:2]) - self.radius)**2)

    def getHMatrix(self, state, ref):
        observation = self.getObservation(state, ref)
        return [-self.weight*(self.centre_pt[0]-state[0])*observation, -self.weight*(self.centre_pt[1]-state[1])*observation, 0, 0, 0]