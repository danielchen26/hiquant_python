import numpy as np

class simulation:

    """This is doc"""
    def __init__(self, S, N, noise):
        self.S = S
        self.N = N
        self.noise = noise

    def generate (self):
        self.K=2
        self.u_true = np.random.rand((self.K,self.N))
