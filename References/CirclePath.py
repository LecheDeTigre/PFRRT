import numpy as np

class CirclePath:
    def __init__(self, centre_pt, radius) -> None:

        self.centre_pt = centre_pt
        self.radius = radius

    def getClosestPoint(self, position):

        heading = np.arctan2(position[1]-self.centre_pt[1], position[0]-self.centre_pt[0])-np.pi/2

        norm_1 = np.linalg.norm(position[0:2]-self.centre_pt)

        distance_along = (np.pi/2-heading)*self.radius

        return np.array([self.centre_pt[0]+self.radius*(position[0]-self.centre_pt[0])/norm_1, self.centre_pt[1]+self.radius*(position[1]-self.centre_pt[1])/norm_1, heading, distance_along])