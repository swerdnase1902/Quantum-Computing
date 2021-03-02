#!/usr/bin/env python
# coding: utf-8

# # QAOA Solver in Cirq
# 
# We will use `cirq` to implement the QAOA solver here.
# 

# First, import the necessary libraries

# In[1]:


import cirq
import numpy as np
from matplotlib import pyplot as plt
import time
import random
import statistics
import sys
import math


# The variable names `n`, `m`, `t`, `C`, `B`, `C`, `B`, `Sep`, and `Mix` follow the notation in lecture notes.
# 
# ![Definition of MaxSAT](./images/Qes](./images/HelperMatrices.png)
# ![QAQA Algorithm](./images/qaoa_alg.png)
# ## Max2SAT Class
# In order to use QAQA, we need to first define the structure of `Max2SAT`

# In[2]:


class Max2SAT:
    def __init__(self, n, m, t, max2sat: str):
        self.n = n
        self.m = m
        self.t = t

        # TODO: discuss how max2sat should look like
        # TODO: initialize and encode a Max2SAT problem into this class
        # self.CNF = ???
        pass

    def Count(self, z) -> int:
        # TODO: implement Count
        return 0

    def Countj(self, j, z) -> int:
        # TODO: implement Countj
        return 0


# ## QAOA Class
# We define a Python class called `QAOASolver` that approximately solves an instance of the MaxSAT problem.

# In[6]:


class QAOASolver:
    def _compute_C_(self):
        # TODO: implement C
        """
        The plan is to enumerate all possible bit strings z and generate the C matrix
        """
        num_qubits = self.n
        C_column_vecs = list()
        num_dim = 2 ** num_qubits
        for z in range(num_dim):
            # Calculate Count(z) * |z>
            Count = self.max2sat.Count(z)
            z_vec = np.zeros(shape=(num_dim, 1))
            z_vec[z, 0] = 1
            C_column_vecs.append(Count * z_vec)
        C = np.concatenate(C_column_vecs, axis=1)
        return np.eye(num_dim)

    def _compute_B(self):
        # TODO: implement B
        num_dim = 2 ** self.n
        B = np.zeros(shape=(num_dim, num_dim))
        NOT = np.array([[0, 1], [1, 0]])
        for k in range(self.n):
            B = B + np.kron(np.eye(2 ** k), np.kron(NOT, np.eye(2 ** (self.n - k - 1))))
        return B

    def __init__(self, max2sat_instance: Max2SAT, num_tries):
        # num_tries is the number of different choices of (gamma, beta)
        self.max2sat = max2sat_instance
        self.num_tries = num_tries
        self.n = max2sat_instance.n
        self.m = max2sat_instance.m
        self.t = max2sat_instance.t

        # TODO: Get C as np.array
        self.C = self._compute_C_()

        # TODO: Get B as np.array
        self.B = self._compute_B()

    def _Mix(self, beta):
        # TODO: implement Mix
        B = self.B
        Mix = np.exp(-1j * beta * B)
        return Mix

    def _Sep(self, gamma):
        # TODO: implement Mix
        C = self.C
        Sep = np.exp(-1j * gamma * C)
        return Sep

    def _make_qaoa_circuit(self, beta, gamma):
        # TODO: implement the circuit
        qubits = cirq.LineQubit.range(self.n)
        ops = [cirq.H(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
        qaoa_circuit = cirq.Circuit(ops)
        return qaoa_circuit

    def solve(self):
        history = list()
        for trial in range(self.num_tries):
            gamma = random.uniform(0, 2 * math.pi)
            beta = random.uniform(0, math.pi)
            circuit = self._make_qaoa_circuit(beta, gamma)
            simulator = cirq.Simulator()
            result = simulator.run(circuit)
            z = result.measurements['result']
            history.append((z, self.max2sat.Count(z)))
        # Pick the measurement z that maximizes Count(z)
        max_z, _ = max(history, key=lambda x: x[1])
        return max_z


# ## Example driver for the above code

# In[9]:


if __name__ == '__main__':
    my_max2sat = Max2SAT(4, 2, 2, "Hello 2SAT")
    solver = QAOASolver(my_max2sat, num_tries=10)
    result = solver.solve()
    # After we turn result into a decimal int, it will represent the maximum number of satisfiable clauses in 2SAT
    result = result.flatten().tolist()
    result.reverse()
    result_int = 0
    while result:
        result_int <<= 1
        result_int |= result.pop()
    print(result_int)
