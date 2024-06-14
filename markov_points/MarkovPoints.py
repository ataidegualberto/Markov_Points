import numpy as np
from scipy import sparse

eps = np.finfo(float).eps

class MarkovPoints:
    def __init__(self,
                 P: np.ndarray | sparse.coo_matrix,
                 P0: np.ndarray,
                 dim: int,
                 neg_sampling: bool = True,
                 max_iter: int = 1000,
                 lr: float = 0.05
                ) -> None:
        
        self.P = P
        self.P0 = P0
        self.is_P_sparse = isinstance(P, sparse.coo_matrix)
        self.dim = dim
        self.neg_sampling = neg_sampling
        self.N = len(P0)
        self.x = np.random.normal(loc=0, scale=0.01, size=(self.N, self.dim))
        self.xtil = np.random.normal(loc=0, scale=0.01, size=(self.N, self.dim))
        self.max_iter = max_iter
        self.lr = lr
        self._J = None
        self._min_J = None
        self._should_update_J = False
    
    @property
    def min_J(self) -> float:
        if self._min_J is None:
            self._min_J = self.get_entropy()
        
        return self._min_J

    @property
    def J(self) -> float:
        if self._J is None or self._should_update_J:
            self._J = self.get_entropy(cross_entropy=True)
        
        self._should_update_J = False
        return self._J

    def fit(self):
        self._original_implementation()
        self._should_update_J = True

    def _original_implementation(self):
        if self.neg_sampling:
            for _ in range(self.max_iter):
                m = np.random.randint(self.N)
                P, neighbor_idxes = self.get_line_of_P(m)
                k_non_neighbors = np.min([self.N, 3*len(P)])
                non_neighbor_idxes = np.setdiff1d(np.arange(self.N), neighbor_idxes)[:k_non_neighbors]
                Q0 = self.calculate_Q0()
                Q = self.calculate_Q(m) # Just one line of Q
                
                w0 = self.P0[m] - Q0[m]
                v0 = self.x[m, :]
                u0 = v0 / np.linalg.norm(v0)

                dJ_dx = np.zeros_like(self.x[m, :])
                for i, n in enumerate(neighbor_idxes):
                    w = P[i] - Q[n]
                    v = self.x[m, :] - self.xtil[n, :]
                    u = v / np.linalg.norm(v)
                    
                    dJ_dx += w * u
                
                dJ_dxtil = -dJ_dx
                dJ_dx += w0 * u0

                repulsion_x = np.zeros_like(dJ_dx)
                for n in non_neighbor_idxes:
                    v = self.x[m, :] - self.xtil[n, :]
                    u = v / np.linalg.norm(v)

                    repulsion_x += Q[n] * u
                
                repulsion_xtil = -repulsion_x

                self.x[m, :] -= self.lr * (dJ_dx - repulsion_x)
                self.xtil[m, :] -= self.lr * (dJ_dxtil - repulsion_xtil)


        else:
            for _ in range(self.max_iter):
                m = np.random.randint(self.N)
                P, idxes = self.get_line_of_P(m)
                Q0 = self.calculate_Q0()
                Q = self.calculate_Q(m)[idxes] # Just one line of Q
                

                w0 = self.P0[m] - Q0[m]
                v0 = self.x[m, :]
                u0 = v0 / np.linalg.norm(v0)

                dJ_dx = w0 * u0
                dJ_dxtil = np.zeros_like(dJ_dx)
                for n in range(idxes.shape[0]):
                    w = P[n] - Q[n]
                    v = self.x[m, :] - self.xtil[n, :]
                    u = v / np.linalg.norm(v)
                    
                    dJ_dx += w * u
                    dJ_dxtil -= w * u

                self.x[m, :] -= self.lr * dJ_dx
                self.xtil[m, :] -= self.lr * dJ_dxtil
        

    def calculate_Q0(self):
        Q0 = np.exp(-np.sqrt(np.sum(self.x ** 2, axis=1)))
        return Q0 / np.sum(Q0)
    
    def calculate_Q(self, idx: int):
        diff = self.x[idx, :].reshape(1, -1) - self.xtil
        d = np.sqrt(np.sum(diff**2, axis=1))
        Q = np.exp(-d)
        
        return Q / np.sum(Q)
    
    def get_line_of_P(self, m):
        if self.is_P_sparse:
            row_mask = self.P.row == m
            probabilities = self.P.data[row_mask]
            indices = self.P.col[row_mask]
        else:
            probabilities = self.P[m, :]
            indices = np.arange(self.N)

        return probabilities, indices


    def _tree_based_implementation():
        pass
    

    def get_entropy(self, cross_entropy: bool = False):
        H = 0.0
        if cross_entropy:
            for i in range(self.N):
                P, idxes = self.get_line_of_P(i)
                Q = self.calculate_Q(i)
                H -= np.sum(P * np.log2(Q[idxes] + eps))
        else:
            for i in range(self.N):
                P, _ = self.get_line_of_P(i)
                H -= np.sum(P * np.log2(P + eps))
        
        return H


        
def intrinsic_dimension(P):
    if isinstance(P, sparse.coo_matrix):
        ID = np.zeros(P.shape[0])

        for i in range(P.shape[0]):
            row_mask = P.row == i
            non_zero_probabilities = P.data[row_mask]  # Usando P.data aqui
            H = -np.sum(non_zero_probabilities * np.log2(non_zero_probabilities + eps))
            ID[i] = 2**H

        return round(np.mean(ID))
    elif isinstance(P, np.ndarray):
        ID = np.zeros(len(P))

        for i in range(len(P)):
            
            H = -np.sum(P[i, :] * np.log2(P[i, :] + eps))
            ID[i] = 2**H
        
        return round(np.mean(ID))
    else:
        raise TypeError("The stochastic matrix P should be either a 2d numpy array or a scipy sparse coo matrix.")

if __name__ == "__main__":
    P = np.array([[0, 0.8, 0.2], [0.5, 0.5, 0], [0.5, 0.4, 0.1]])
    P0 = np.array([9/27, 16/27, 2/27])

    mkpts = MarkovPoints(P, P0, 2, False)
    print(intrinsic_dimension(P))
    mkpts.fit()

    print(mkpts.calculate_Q0())
    for i in range(3):
        print(mkpts.calculate_Q(i))