import numpy as np


class Discrete:

    def __init__(self, pmat: np.ndarray):
        shape = pmat.shape

        if shape[0] != shape[1]:
            raise ValueError("Markov chain matrix is not square")
        if not np.all(np.logical_and(0 <= pmat, pmat <= 1)):
            raise ValueError("Markov chain matrix does not consist of probabilities")
        rowsums = np.sum(pmat, axis=1)
        if not np.all(rowsums == 1):
            raise ValueError("Markov chain matrix rows don't sum to 1")
        # Square matrix with probability like entries with rows that sum to 1
        # Looks like a reasonable markov chain.

        self.pmat = pmat

    def nstep(self, steps):
        return np.linalg.matrix_power(self.pmat, steps)

    def hittime(self, states):
        tempmat = self.pmat

        tempmat = np.delete(tempmat, states, 0)
        tempmat = np.delete(tempmat, states, 1)
        P = tempmat
        dim = P.shape[0]
        solved = np.linalg.solve(np.eye(dim) - P, np.ones(dim))
        for state in states:
            solved = np.insert(solved, state, 0)
        return solved

    def hitprob(self, states):
        tempmat = self.pmat
        tempmat = np.delete(tempmat, states, 0)
        rowsrem = tempmat
        tempmat = np.delete(tempmat, states, 1)
        P = tempmat
        #Deprecated
        q = np.sum((rowsrem.transpose()[[states]]).transpose(), axis=1)
        dim = P.shape[0]
        solved = np.linalg.solve(np.eye(dim) - P, q)
        for state in states:
            solved = np.insert(solved, state, 1)
        return solved


if __name__ == '__main__':
    mc = Discrete(np.array([[1 / 2, 1 / 5, 3 / 10, 0],
                            [1 / 5, 3 / 10, 2 / 5, 1 / 10],
                            [1 / 10, 2 / 5, 2 / 5, 1 / 10],
                            [0, 1 / 10, 3 / 5, 3 / 10]]))
    print(mc.nstep(7))
    print(mc.hittime([1, 3]))
    print(mc.hitprob([1, 3]))
