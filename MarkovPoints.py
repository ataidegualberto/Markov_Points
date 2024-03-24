import numpy as np

eps = np.finfo(float).eps

class MarkovPoints:
    def __init__(self,
                 P: np.ndarray,
                 P0: np.ndarray,
                 dim: int,
                 neg_sampling: bool = True,
                ) -> None:
        
        self.P = P
        self.P0 = P0
        self.dim = dim
        self.neg_sampling = neg_sampling
        self.N = len(P0)
        self.x = np.random.normal(loc=0, scale=0.01, size=(self.N, self.dim))
        self.xtil = np.random.normal(loc=0, scale=0.01, size=(self.N, self.dim))
        self.n_iter = 1000
        self.alfa = 0.1
        self.J = 0.0
    
    def fit(self):
        self._original_implementation()

    def _original_implementation(self):
        for _ in range(self.n_iter):
            for m in range(self.N):
                Q0 = self.calculate_Q0()
                Q = self.calculate_Q(m) # Just one line of Q

                w0 = self.P0[m] - Q0[m]
                v0 = self.x[m, :]
                u0 = v0 / np.sqrt(np.sum(v0 ** 2))

                dJ_dx = w0 * u0
                dJ_dxtil = np.zeros_like(dJ_dx)
                for n in range(self.N):
                    w = self.P[m, n] - Q[n]
                    v = self.x[m, :] - self.xtil[n, :]
                    u = v /np.sqrt(np.sum(v ** 2))
                    
                    dJ_dx += w * u
                    dJ_dxtil -= w * u

                self.x[m, :] -= self.alfa * dJ_dx
                self.xtil[m, :] -= self.alfa * dJ_dxtil
        

    def calculate_Q0(self):
        Q0 = np.exp(-np.sqrt(np.sum(self.x ** 2, axis=1)))
        return Q0 / np.sum(Q0)
    
    def calculate_Q(self, idx: int):
        diff = self.x[idx, :].reshape(1, -1) - self.xtil
        d = np.sqrt(np.sum(diff**2, axis=1))
        Q = np.exp(-d)
        
        return Q / np.sum(Q)

    def _tree_based_implementation():
        pass

    def intrinsic_dimension(self):
        ID = np.zeros_like(self.P0)

        for i in range(len(ID)):
            H = -np.sum(self.P[i, :] * np.log2(self.P[i, :] + eps))
            ID[i] = 2**H
        
        return round(np.mean(ID))

        


if __name__ == "__main__":
    P = np.array([[0, 0.8, 0.2], [0.5, 0.5, 0], [0.5, 0.4, 0.1]])
    P0 = np.array([9/27, 16/27, 2/27])

    mkpts = MarkovPoints(P, P0, 2, False)
    print(mkpts.intrinsic_dimension())
    mkpts.fit()

    print(mkpts.calculate_Q0())
    for i in range(3):
        print(mkpts.calculate_Q(i))