import numpy as np
import itertools

from Models.Model import Model

class DubinsCarSpatial(Model):

    def __init__(self) -> None:
        super().__init__()

        self.L = 5.

        self.N_states = 3
        self.N_actuations = 1

    def f(self, x, u):
    
        psi = x[2]
        
        delta = u[0]

        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta)/self.L])

        return dx

    def step(self, x0, u, ds):

        k1 = self.f(x0, u)
        k2 = self.f(x0+ds*k1/2, u)
        k3 = self.f(x0+ds*k2/2, u)
        k4 = self.f(x0+ds*k3, u)

        curr_state = x0 + ds*(k1+2*k2+2*k3+k4)/6

        return curr_state

    def sampleRandomInput(self, prev_state=None, ds=None, num_samples=1):

        delta = np.random.normal(0, np.pi/2, num_samples)
        # import pdb; pdb.set_trace()
        delta[0] = np.array([min(max(delta[0], -np.pi), np.pi)])

        return delta
    
    def getActuationSet(self):
        delta = [-np.pi/4, 0., np.pi/4]

        return delta