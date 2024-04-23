from abc import ABC, abstractmethod

class SearchProblem(ABC):
    
    @abstractmethod
    def StateCost(self, observation, desired_observation):
        pass
    
    @abstractmethod
    def EstimatedCostToGoal(self, optimal_state, goal_state):
        pass
    
    @abstractmethod
    def InvalidState(self, observation):
        pass
        
    @abstractmethod
    def InGoalSet(self, state):
        pass