import numpy as np
import itertools

from Models.Model import Model

class DubinsCar(Model):

    def __init__(self, R) -> None:
        super().__init__()

        self.L = 5.

        self.N_states = 5
        self.N_actuations = 2

        self.R = R

    def f(self, x, u):
    
        psi, delta = x[2], x[3]
        
        u1, u2 = u[0], u[1]

        dx = np.array([np.cos(psi), np.sin(psi), np.tan(delta)/self.L, u1, u2])

        return dx

    def step(self, x0, u, dt):

        k1 = self.f(x0, u)
        k2 = self.f(x0+dt*k1/2, u)
        k3 = self.f(x0+dt*k2/2, u)
        k4 = self.f(x0+dt*k3, u)

        curr_state = x0 + dt*(k1+2*k2+2*k3+k4)/6

        return curr_state

    def sampleRandomInput(self, prev_state, dt):

        delta = prev_state[3]

        steering_rate_lim = 16*np.pi

        steering_lim = 3*np.pi/8

        # import pdb; pdb.set_trace()

        (u1, u2) = np.random.multivariate_normal([0, 0], self.R, (1,))[0,:]

        effective_steering_lim_max = min(steering_rate_lim, (steering_lim-delta)/dt)
        effective_steering_lim_min = max(-steering_rate_lim, (-steering_lim-delta)/dt)

        u1 = np.array(min(max(u1, effective_steering_lim_min), effective_steering_lim_max))
        u2 = np.array(min(max(u2, -2), 2))

        return np.array([u1, u2])
    
    def getActuationSet(self):
        u1 = [-np.pi/6, 0., np.pi/6]
        u2 = [-2, 2]

        return np.array([u1, u2])