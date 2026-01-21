from abc import ABC, abstractmethod

class SummaryBase(ABC):
    @abstractmethod
    def generate(self, documents):
        pass
