from abc import ABC, abstractmethod

class SequenceSampler(ABC):

    def __init__(self, model):
        self.model = model

    @abstractmethod
    def sample(self, max_length):
        pass