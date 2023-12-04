from abc import ABC, abstractmethod

class Model(ABC):
    
    @abstractmethod
    def step(u, t):
        pass
    
    @abstractmethod
    def sampleRandomInput(self, prev_state=None):
        pass