import numpy as np
from cvxopt import solvers, matrix
from scipy.linalg import block_diag
import sys


def SVD(S, N, X):
    """SVD sovler."""
    A1 = np.kron(S, np.eye(N))
    A2 = block_diag(*X).T
    A = np.concatenate((A1, A2), axis=1)
    U, s, V = np.linalg.svd(A, full_matrices=True)
    return U, s, V


def QP(S, N, X):
    """Quadratic Programming Solver."""
    A1 = np.kron(S, np.eye(N))
    A2 = block_diag(*X).T
    A1 = matrix(A1)
    A2 = matrix(A2)
    AA = matrix([[A1], [A2]])
    P = matrix(np.dot(AA.T, AA))
    dim = P.size[0]
    q = matrix(np.zeros(dim))
    G = matrix(-np.eye(dim))
    h = matrix(np.zeros(dim), (dim, 1))
    A = matrix(np.ones(dim), (1, dim))
    b = matrix(1.0)
    sol = solvers.qp(P, q, G, h, A, b)['x']
    return sol
    # print(sol['x'])


def LP(S, N, X):
    """Linear Programing Solver."""

    return
    
def CD(S, X, iter_num):
    """Coordinate Descent Solver"""
    if len(sys.argv) < 3:
        iter_num = 1000
    Num_pep, Num_prot = S.shape
    Num_condition = X.shape[1]
    
    # for loop iterations
    for 
    return protein, opt_Z, opt_rsq, opt_model
# S = [[1,0],[1,1]]
# N = 6
# X = np.random.rand(2, N)
# QP(S, N, X)
