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
import scipy
import argparse


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

        # For a problem looking like (x_1 or x_2) and  (x_2 or not x_3):
        # It will be represented as :
        # (0, 1, 0, 0), (1, 2, 0, 0)
        self.clauses = self.random_generate()
        print(self)

    def Count(self, z) -> int:
        # TODO: implement Count
        # z is either an int or a list of zeros and ones
        if isinstance(z, int):
            z = bin(z)[2:].zfill(self.n)
            z = [[int(i) for i in z]]

        z = z[0]

        total = 0
        for j in range(self.m):
            total += self.Countj(j, z)

        return total

    def Countj(self, j, z) -> int:
        clause = self.clauses[j]
        v_1, v_2, v_1_negate, v_2_negate = clause

        if (v_1_negate == 1 and z[v_1] == 0) or (v_1_negate == 0 and z[v_1] == 1) \
                or (v_2_negate == 1 and z[v_2] == 0) or (v_2_negate == 0 and z[v_2] == 1):
            return 1
        else:
            return 0

    def random_generate(self):
        current_clauses = set()
        while True:
            v_1 = random.randrange(0, self.n - 1)  # Sample the variable with smaller index
            v_1_negate = random.randrange(0, 2)  # Whether variable_1 is negated or not
            v_2 = random.randrange(v_1 + 1, self.n)  # Sample the variable with larger index
            v_2_negate = random.randrange(0, 2)  # Whether variable_2 is negated or not
            clause_tuple = (v_1, v_2, v_1_negate, v_2_negate)
            current_clauses.add(clause_tuple)
            if len(current_clauses) == self.m:
                break
        return list(current_clauses)

    def __str__(self):
        report_str = []
        for clause in self.clauses:
            report_str.append(
                "({}v_{} OR {}v_{})".format("~" if clause[2] else "", clause[0], "~" if clause[3] else "", clause[1]))

        return "  AND  ".join(report_str)


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
        return C  # np.eye(num_dim)

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
        Mix = -1j * beta * B
        Mix = scipy.linalg.expm(Mix)
        return Mix

    def _Sep(self, gamma):
        # TODO: implement Mix
        C = self.C
        Sep = -1j * gamma * C
        Sep = scipy.linalg.expm(Sep)
        return Sep

    def _make_qaoa_circuit(self, beta, gamma):

        # Initializing the qubits
        n = self.n
        inputs = [cirq.GridQubit(i, 0) for i in range(n)]
        circuit = cirq.Circuit()

        # 1. Apply H^N to the input quibuts
        for i in range(n):
            circuit.append(cirq.H(inputs[i]))

        # 2. Add Sep(gamma)
        circuit.append(cirq.ops.MatrixGate(self._Sep(gamma))(*inputs))

        # 3. Add Mix(Betta)
        circuit.append(cirq.ops.MatrixGate(self._Mix(beta))(*inputs))

        # 4. Measurement
        circuit.append(cirq.measure(*inputs, key='result'))

        return circuit

        '''qubits = cirq.LineQubit.range(self.n)
        ops = [cirq.H(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
        qaoa_circuit = cirq.Circuit(ops)
        return qaoa_circuit'''

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
        max_z, num_clause = max(history, key=lambda x: x[1])
        return max_z, num_clause


# ## Example driver for the above code

# In[9]:

def run_benchmark():
    max_n = 14
    print('We will study how n, the number of variables, affects the execution time of QAOA')
    print('We will vary n from {} to {}'.format(2, max_n))
    print('Begin benchmarking:')
    n_list = list()
    exec_times = list()
    for n in range(2, max_n):
        m = 2 * n
        print('Testing n={} and m={}'.format(n, m))
        start = time.time()
        max2sat = Max2SAT(n, m, 2, "Not Used")
        print('The Max2SAT instance that we want to solve by QAQA is {}'.format(max2sat))
        solver = QAOASolver(max2sat, num_tries=1)
        result, num_clause = solver.solve()
        end = time.time()
        elapsed = end - start
        print('It took {} seconds for QAQA to solve'.format(elapsed))
        n_list.append(n)
        exec_times.append(elapsed)
        print("")

    print('Here\'s the graph that tells you how n and execution times relate')
    plt.scatter(n_list, exec_times)
    plt.plot(n_list, exec_times)
    plt.xlabel('n: number of variables')
    plt.ylabel('execution time')
    plt.title('Number of Variables vs Execution Time')
    plt.show()


def run_custom_input(n, m, sat_str):
    my_max2sat = Max2SAT(n, m, 2, sat_str)
    print('We will run QAQA with your custom input {}'.format(my_max2sat))
    solver = QAOASolver(my_max2sat, num_tries=10)
    result, num_clause = solver.solve()

    print("The variable assignment we found: ", result)
    print("Max number of clause that can be satisfied:", num_clause)


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Demonstration of the QAQA algorithm')
    argparser.add_argument('-b', '--benchmark', help='Run various benchmark', action='store_true', default=True)
    argparser.add_argument('-c', '--custom2SAT', action='store_true',
                           help='Run QAQA algorithm with custom 2SAT instance',
                           default=False)
    argparser.add_argument('-s', '--str_repr',
                           help='If --custom2SAT is set, set the 2SAT string representation')
    argparser.add_argument('-n', '--num_vars',
                           help='If --custom2SAT is set, set the number of variables of the 2SAT instance')
    argparser.add_argument('-m', '--num_clauses',
                           help='If --custom2SAT is set, set the number of clauses of the 2SAT instance')

    args = vars(argparser.parse_args())

    if args['custom2SAT']:
        n = int(args['num_vars'])
        m = int(args['num_clauses'])
        twoSAT_str = args['str_repr']
        run_custom_input(n, m, twoSAT_str)
    elif args['benchmark']:
        run_benchmark()
    else:
        print('Error: You need to either provide --custom2SAT option or --benchmark option. Exiting',
              file=sys.stderr)
        exit(1)


