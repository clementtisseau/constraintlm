from .base import SequenceSampler


class BeamSearchSampler(SequenceSampler):

    def __init__(self, model, beta, constraint = None):
        super().__init__(model, constraint)
        self.beta = beta

    def sample(self, max_length, num_sentences):
        pass