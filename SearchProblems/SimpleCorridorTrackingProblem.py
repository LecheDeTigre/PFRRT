import numpy as np

from SearchProblems.SearchProblem import SearchProblem

class SimpleCorridorTrackingProblem(SearchProblem):
    def __init__(self, goal_state, GetClosestPoint):
        self.goal_state = goal_state
        self.GetClosestPoint = GetClosestPoint
    
    def StateCost(self, observation, desired_observation):
        return np.matmul(observation-desired_observation, np.matmul(np.diag([5, 2, 1]), np.transpose(observation-desired_observation)))
    
    def EstimatedCostToGoal(self, optimal_state, goal_state):
        optimal_state_projection = self.GetClosestPoint(optimal_state)
        goal_state_projection = self.GetClosestPoint(goal_state)

        x, y, psi, s = optimal_state_projection[0], optimal_state_projection[1], optimal_state_projection[2], optimal_state_projection[3]
        x_ref, y_ref, psi_ref, s_ref = goal_state_projection[0], goal_state_projection[1], goal_state_projection[2], goal_state_projection[3]

        eY_ref = -(goal_state[0]-x_ref)*np.sin(psi_ref) + (goal_state[1]-y_ref)*np.cos(psi_ref)
        eY = -(optimal_state[0]-x)*np.sin(psi) + (optimal_state[1]-y)*np.cos(psi)
        
        return 1.2* ((eY-eY_ref)**2 + (s-s_ref)**2)
    
    def InvalidState(self, observation):
        # import pdb; pdb.set_trace()
        y_lim = 0.20
        psi_lim = np.pi/3

        if (-y_lim <= observation[1] and observation[1] <= y_lim):
            return False
        else:
            return True
        
    def InGoalSet(self, state):
        if(np.linalg.norm(state[0:2]-self.goal_state[0:2]) <= 0.25) and abs(state[2]-self.goal_state[2]) <= np.pi/12:
            return True
        else:
            return False