import numpy as np

class MarkovPoints:
    def __init__(self, P: np.ndarray, P0: np.ndarray, dim: int) -> None:
        self.P = P
        self.P0 = P0
        self.dim = dim
    
    def fit(self):
        pass