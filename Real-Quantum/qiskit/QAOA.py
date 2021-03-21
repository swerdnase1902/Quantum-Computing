
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
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit import IBMQ, assemble, transpile
from typing import Tuple
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
    return "9da9d4d05ae90bc8114510e18006eb1fc5435fb057c441ca9be7a83a59c4b864c16b1e51a91efc772e1b000c655a84a267708418bea1852a129af8fdfad31e68"

def send_to_ibm(circuit: cirq.Circuit, repetitions):
    circ = circuit
    qbits = circ.all_qubits()

    provider = IBMQ.load_account()
    # backend = provider.backends.backend_name
    backend = provider.backends.ibmq_athens

    def cirq_to_qiskit(circuit: 'cirq.Circuit', qubits: Tuple['cirq.Qid', ...]):
        # here you go: https://quantumcomputing.stackexchange.com/questions/14164/can-i-run-cirq-on-ibmq
        qasm_output = cirq.QasmOutput((circuit.all_operations()), qubits)
        qasm_circuit = QuantumCircuit().from_qasm_str(str(qasm_output))
        # Execute the circuit qiskit backend
        # job = execute(qasm_circuit, Aer.get_backend(backend), shots=1000)
        # Grab results from the job
        return qasm_circuit
    qiskit_circuit = cirq_to_qiskit(circ, qbits)


    transpiled = transpile(qiskit_circuit, backend)
    qobj = assemble(transpiled, backend, shots=repetitions)
    job = backend.run(qobj)
    result = job.result()
    counts = result.get_counts()
    delayed_result = backend.retrieve_job(job.job_id()).result()
    delayed_counts = delayed_result.get_counts()
    return delayed_counts

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
        # We are hardcoding the example (x0 OR x1) in lecture with 2 clauses. Following https://piazza.com/class/kjky6kvh4v21rm?cid=385, we need four qubits
        n = 4
        actual_n = 2

        inputs = [cirq.GridQubit(i, 0) for i in range(n)]
        circuit = cirq.Circuit()

        # 1. Apply H^N to the input quibuts, don't apply to helper qubit
        for i in range(actual_n):
            circuit.append(cirq.H(inputs[i]))

        
        # helper qubit should be set to 1
        circuit.append(cirq.X(inputs[-1]))

        for i in range(actual_n):
            circuit.append(cirq.X(inputs[i]))
        circuit.append(cirq.TOFFOLI(inputs[0], inputs[1], inputs[3]))

        circuit.append(cirq.X(inputs[3]))

        circuit.append(cirq.CZPowGate(exponent=-4)(inputs[3], inputs[2]))
        
        circuit.append(cirq.TOFFOLI(inputs[0], inputs[1], inputs[3]))

        # 3. Add Mix(Beta) We are hardcoding Mixer with beta=pi/2. Make sure don't touch the helper qubit
        # circuit.append(cirq.ops.MatrixGate(self._Mix(beta))(*inputs))
        for i in range(actual_n):
            circuit.append(cirq.X(inputs[i]))

        # 4. Measurement. Don't measure the helper qubit
        circuit.append(cirq.measure(*(inputs[:actual_n]), key='result'))

        return circuit

    def solve(self):
        gamma = np.pi / 4
        # Hardcode beta = pi/2
        beta = np.pi / 2  # random.uniform(0, math.pi)
        circuit = self._make_qaoa_circuit(beta, gamma)

        restuls = send_to_ibm(circuit ,repetitions=50)
        
        history = list()
        for key, value in restuls.items():
            z = np.array([int(i) for i in key])
            history.append((z, self.max2sat.Count(z)))
        max_z, num_clause = max(history, key=lambda x: x[1])
        return max_z, num_clause
    
    def solve_result(self):
        restuls = {
            "11": 20,
            "10": 12,
            "00": 12,
            "01": 6
        }
        history = list()
        for key, value in restuls.items():
            z = [np.array([int(i) for i in key])]
            history.append((z, self.max2sat.Count(z)))
        max_z, num_clause = max(history, key=lambda x: x[1])
        return max_z, num_clause

    def solve_normal(self):
        history = list()
        for trial in range(self.num_tries):
            gamma = np.pi / 4
            # Hardcode beta = pi/2
            beta = np.pi / 2  # random.uniform(0, math.pi)
            circuit = self._make_qaoa_circuit(beta, gamma)
            simulator = cirq.Simulator()
            result = simulator.run(circuit)
            z = result.measurements['result']
            history.append((z, self.max2sat.Count(z)))
        # Pick the measurement z that maximizes Count(z)
        max_z, num_clause = max(history, key=lambda x: x[1])
        return max_z, num_clause

def run_hardcoded_input_on_ibm():
    my_max2sat = Max2SAT(2, 1, 2, "V0 OR V1")
    print('We will run QAQA with hardcoded input (x0 OR x1) on IBM')
    solver = QAOASolver(my_max2sat, num_tries=10)
    print("Normal solver : ", solver.solve_normal()[1])
    #solver.solve()
    print("IBM", solver.solve())


if __name__ == '__main__':

    run_hardcoded_input_on_ibm()


