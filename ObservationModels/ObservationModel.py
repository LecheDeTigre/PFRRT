from abc import ABC, abstractmethod

class ObservationModel(ABC):
    @abstractmethod
    def getObservation(self, state, ref):
        pass
    
    @abstractmethod
    def getHMatrix(self, state, ref):
        pass