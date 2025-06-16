from abc  import ABC, abstractmethod

class BaseLM(ABC):

    @abstractmethod
    def logits():
        pass