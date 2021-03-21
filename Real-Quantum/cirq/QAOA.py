
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
import requests

class Max2SAT:
    def __init__(self, n, m, t, max2sat = None):
        self.n = n
        self.m = m
        self.t = t

        # For a problem looking like (x_1 or x_2) and  (x_2 or not x_3):
        # It will be represented as :
        # (0, 1, 0, 0), (1, 2, 0, 0)
        if max2sat is None:
            self.clauses = self.random_generate()
            #print(self)
        else:
            self.clauses = self.parse(max2sat)

    def parse(self, max2sat_string):
        #"V1 OR V2 AND V2 OR ~V3"
        max2sat_string = max2sat_string.lower()
        max2sat_string = max2sat_string.replace(" ", "")
        clauses = []
        for clause in max2sat_string.split("and"):
            v_a_str = clause.split("or")[0]
            v_b_str = clause.split("or")[-1]
            if v_a_str[0] == "~":
                v_a_negate = 1
            else:
                v_a_negate = 0
            v_a = int(v_a_str.strip("~").strip("v"))

            if v_b_str[0] == "~":
                v_b_negate = 1
            else:
                v_b_negate = 0
            v_b = int(v_b_str.strip("~").strip("v"))
            clauses.append((v_a, v_b, v_a_negate, v_b_negate))
        return clauses

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
    
    def exact_solve(self):
        num_qubits = self.n
        max_count = 0
        num_dim = 2 ** num_qubits
        for z in range(num_dim):
            Count = self.Count(z)
            if Count > max_count:
                max_count = Count
        return max_count

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


def load_credential():
    # will return (ucla_email, student_id)
    if hasattr(load_credential, 'email') and hasattr(load_credential, 'uid'):
        return load_credential.email, load_credential.uid
    credential_file = open('../cirq_credentials.txt', 'r')
    lines = credential_file.readlines()
    credential_file.close()
    email = lines[0].strip()
    uid = lines[1].strip()

    load_credential.email = email
    load_credential.uid = uid
    return email, uid

def send_to_google(circuit: cirq.Circuit, repetitions=100):
    send_url = 'http://quant-edu-scalability-tools.wl.r.appspot.com/send'
    json_circuit = cirq.to_json(circuit)
    email, uid = load_credential()
    job_payload = {
        'circuit': json_circuit,
        'email': email,
        'repetitions': repetitions,
        'student_id': uid
    }
    response = requests.post(url=send_url, json=job_payload)
    return response
    print(response.text)
class QAOASolver:
    def _compute_C_(self):
        # Since we are hard-coding, don't even need this
        pass

    def _compute_B(self):
        # Since we are hard-coding, don't even need this
        pass

    def __init__(self, max2sat_instance: Max2SAT, num_tries):
        # Probably don't need the following 2 lines of code since we are hardcoding (x0 AND (x0 OR x1))
        # num_tries is the number of different choices of (gamma, beta)
        self.max2sat = max2sat_instance
        self.n = max2sat_instance.n
        self.num_tries = num_tries

        # No need of the following two lines since we are hardcoding x0 AND (x0 OR x1)
        # self.m = max2sat_instance.m
        # self.t = max2sat_instance.t

        # TODO: Get C as np.array
        self.C = self._compute_C_()

        # TODO: Get B as np.array
        self.B = self._compute_B()

    def _Mix(self, beta):
        # Hardcoding the Mixer Gate, so we don't even need this function
        pass

    def _Sep(self, gamma):
        # Hardcoding the Mixer Gate, so we don't even need this function
        pass

    def _make_qaoa_circuit(self, beta, gamma):

        # Initializing the qubits
        # We are hardcoding the example (x0 AND (x0 OR x1)) in lecture with 2 clauses. Since the Sep requires 1 extra helper qubits, we set n=3 here
        n = 3

        inputs = [cirq.GridQubit(i, 0) for i in range(n)]
        circuit = cirq.Circuit()

        # 1. Apply H^N to the input quibuts, don't apply to helper qubit
        for i in range(n-1):
            circuit.append(cirq.H(inputs[i]))
        # helper qubit should be set to 1
        circuit.append(cirq.X(inputs[-1]))

        # 2. Add Sep(gamma)
        # TODO: Harold
        circuit#.append(???)


        # 3. Add Mix(Beta) We are hardcoding Mixer with beta=pi/2. Make sure don't touch the helper qubit
        # circuit.append(cirq.ops.MatrixGate(self._Mix(beta))(*inputs))
        for i in range(n-1):
            circuit.append(cirq.X(inputs[i]))

        # 4. Measurement. Don't measure the helper qubit
        circuit.append(cirq.measure(*(inputs[:-1]), key='result'))

        return circuit

        '''qubits = cirq.LineQubit.range(self.n)
        ops = [cirq.H(q) for q in qubits] + [cirq.measure(*qubits, key='result')]
        qaoa_circuit = cirq.Circuit(ops)
        return qaoa_circuit'''

    def solve(self):
        gamma = random.uniform(0, 2 * math.pi)
        # Hardcode beta = pi/2
        beta = np.pi / 2  # random.uniform(0, math.pi)
        circuit = self._make_qaoa_circuit(beta, gamma)

        response = send_to_google(circuit)
        print('Just sent your circuit to Google, below is the response.text')
        print(response.text)
        # parse job_id
        jobid = response.text[response.text.find(':') + 1:].strip()
        return jobid

        # simulator = cirq.Simulator()
        # result = simulator.run(circuit)
        # z = result.measurements['result']

# def test_correctness():
#     from tqdm import tqdm
#     errors = []
#     for n in range(3, 8):
#         for m in range(2, n * 3):
#             my_max2sat = Max2SAT(n, m, 2)
#             solver = QAOASolver(my_max2sat, num_tries=10)
#             result, num_clause = solver.solve()
#
#             errors.append(abs(my_max2sat.exact_solve() - num_clause))
#     errors = np.array(errors)
#     print("Out of {} random test cases, the algorithm is completely correct for {} of the cases.".format(len(errors),
#                                                                                                          int(sum((
#                                                                                                                              errors == 0)))))
#     print("Average difference with respect to the correct answer {}".format(float(sum(errors) / errors.shape[0])))

# def run_benchmark():
#     max_n = 14
#     print('We will study how n, the number of variables, affects the execution time of QAOA')
#     print('We will vary n from {} to {}'.format(2, max_n))
#     print('Begin benchmarking:')
#     n_list = list()
#     exec_times = list()
#     for n in range(2, max_n):
#         m = 2 * n
#         print('Testing n={} and m={}'.format(n, m))
#         start = time.time()
#         max2sat = Max2SAT(n, m, 2)
#         print('The Max2SAT instance that we want to solve by QAQA is {}'.format(max2sat))
#         solver = QAOASolver(max2sat, num_tries=1)
#         result, num_clause = solver.solve()
#         end = time.time()
#         elapsed = end - start
#         print('It took {} seconds for QAQA to solve'.format(elapsed))
#         n_list.append(n)
#         exec_times.append(elapsed)
#         print("")
#
#     print('Here\'s the graph that tells you how n and execution times relate')
#     plt.scatter(n_list, exec_times)
#     plt.plot(n_list, exec_times)
#     plt.xlabel('n: number of variables')
#     plt.ylabel('execution time')
#     plt.title('Number of Variables vs Execution Time')
#     plt.show()


def run_hardcoded_input_on_google():
    # TODO: Harold hardcode my_max2sat to encode (x0 AND (x0 OR x1))
    my_max2sat = Max2SAT(1, 2, 2, '')
    print('We will run QAQA with hardcoded input (x0 AND (x0 OR x1)) on Google')
    solver = QAOASolver(my_max2sat, num_tries=10)
    solver.solve()

    # print("The variable assignment we found: ")
    # for index, i in enumerate(result[0]):
    #     print("  v_{} : {}".format(index, i))
    # print("Max number of clause that can be satisfied:", num_clause)

# def test_correctness_wrapper():
#     print("##################### A Random Test Run #####################")
#     my_max2sat = Max2SAT(6, 50, 2)
#     print(my_max2sat)
#     solver = QAOASolver(my_max2sat, num_tries=10)
#     result, num_clause = solver.solve()
#     print("\nThe variable assignment we found: ")
#     for index, i in enumerate(result[0]):
#         print("  v_{} : {}".format(index, i))
#     print("")
#     print("Max number of clause that can be satisfied:", num_clause)
#     print("Exact solver:", my_max2sat.exact_solve())
#
#     print('\n\n\n')
#     print("##################### Testing Correctness #####################")
#     test_correctness()

if __name__ == '__main__':

    run_hardcoded_input_on_google()

