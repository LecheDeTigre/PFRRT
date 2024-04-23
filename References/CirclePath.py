import numpy as np

from References.Reference import Reference

class CirclePath(Reference):
    def __init__(self, centre_pt, radius) -> None:

        self.centre_pt = centre_pt
        self.radius = radius

    def getClosestPoint(self, state):

        heading = np.arctan2(state[1]-self.centre_pt[1], state[0]-self.centre_pt[0])-np.pi/2

        norm_1 = np.linalg.norm(state[0:2]-self.centre_pt)

        distance_along = (np.pi/2-heading)*self.radius

        return np.array([self.centre_pt[0]+self.radius*(state[0]-self.centre_pt[0])/norm_1, self.centre_pt[1]+self.radius*(state[1]-self.centre_pt[1])/norm_1, heading, distance_along])