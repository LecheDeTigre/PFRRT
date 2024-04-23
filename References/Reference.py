from abc import ABC, abstractmethod

class Reference(ABC):
    
    @abstractmethod
    def getClosestPoint(self, state):
        pass