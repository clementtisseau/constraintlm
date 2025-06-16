from abc import ABC, abstractmethod

class SequenceSampler(ABC):

    def __init__(self, model, constraint = None):
        self.model = model
        self.constraint = constraint

    @abstractmethod
    def sample(self, max_length):
        pass