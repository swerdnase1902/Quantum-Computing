from collections import Counter
import random
import numpy as np
import scipy as sp
import cirq
import time
from pprint import pprint
from tqdm import tqdm
import math
import requests
from qiskit import QuantumCircuit, execute, Aer
from qiskit.visualization import plot_histogram
from qiskit import IBMQ, assemble, transpile
from typing import Tuple
'''
This is how we create the oracle:

1. First we copy the inputs to outputs qubits
2. The current qubit is |x> |x>. We could make it into 
        |x> |x + b>, for half of x; 
        |x> |x>, for the other half.
    Then for any |x_1>|x_1>, |x_2>|x_2> such that x_1 + x_2 = b, they are mapped into either:
        |x_1>|x_1> and |x_2>|x_1>
    or 
        |x_1>|x_2> and |x_2>|x_2>, depending on which half x_1 falls into.

    To determine for any x, which half it belongs to, we use a simple heuristic (which heuristic we choose does not matter as long as it devides the inputs space by half), find a random bit of s such that the bit is 1, if the same bit of x is 1, then we count it as in the first half.

3. Now we only need to add a cirquit that simulates g, which satisfies:
    g(x) != g(y), iff x != y
   This could be done as a random permutation function.
'''
def oracle(inputs, outputs, s, circuit, n, swap_time = 0, not_time = 0): 
    # 1
    for index in range(len(inputs)):
        circuit.append(cirq.CNOT(inputs[index], outputs[index])) # Becuase outputs_quibuts are all zeros, we can do the copy with CNOT

    # 2
    all_non_zero_indexes_of_s = []
    for index, i in enumerate(s):
        if i:
            all_non_zero_indexes_of_s.append(index)
    if len(all_non_zero_indexes_of_s) > 0:
        chosen_index = random.choice(all_non_zero_indexes_of_s)
        # Apply a CNOT only when 1) s's current bit is not zero 2) the inputs x falls into the first half
        for i in range(len(s)):
            if s[i]:
                circuit.append(cirq.CNOT(inputs[chosen_index], outputs[i]))
    
    # 3
    # The random permutation circuits could be very complex or very easy based on our given parameters.
    # We implement the random permutation circuit with a bunch of SWAP and NOT gates.
    
    # Add swap_time random SWAP operations
    for i in range(swap_time):

        # Pick two random positions
        pos_1 = random.randint(0, n - 1)
        while True:
            pos_2 = random.randint(0, n - 1)
            if pos_2 != pos_1:
                break
        
        # SWAP
        circuit.append(cirq.SWAP(outputs[pos_1], outputs[pos_2]))
    
    # Add not_time random NOT operations
    for i in range(not_time):
        # Pick one random positions
        pos = random.randint(0, n - 1)
        circuit.append(cirq.X(outputs[pos]))
    

def linear_solver(matrix, tolerance):
    # Calculate the rank of the results
    U, s, V = np.linalg.svd(matrix)
    rank = np.sum(np.abs(s) > tolerance)

    if rank == n - 1: # If the rank is full
        null_space = sp.linalg.null_space(matrix).T[0]
        solution = np.around(null_space, 3) # Discard small values
        minval = abs(min(solution[np.nonzero(solution)], key=abs))
        solution = (solution / minval % 2).astype(int)
        return True, solution
    else:
        return False, None

def load_credential():
    return "9da9d4d05ae90bc8114510e18006eb1fc5435fb057c441ca9be7a83a59c4b864c16b1e51a91efc772e1b000c655a84a267708418bea1852a129af8fdfad31e68"


def make_circuit(s, n, not_time, swap_time, m, tolerance):

    # Initializing the qubits
    inputs = [cirq.GridQubit(i, 0) for i in range(n)]  # inputs x
    outputs = [ cirq.GridQubit(i + n, 0) for i in range(n)]
    circuit = cirq.Circuit()

    # 1. Apply H^N to the input quibuts
    for i in range(n):
        circuit.append(cirq.H(inputs[i]))

    # 2. Add U_f
    oracle(inputs, outputs, s, circuit, n, swap_time=swap_time, not_time=not_time)

    # 3. Apply H^N to the inputs qubits
    for i in range(n):
        circuit.append(cirq.H(inputs[i]))

    # 4. Measurement
    circuit.append(cirq.measure(*inputs, key='result'))

    #print("Randomly constructed circuit:")
    #print(circuit)
    return circuit

def make_a_run_ibm(n, not_time, swap_time, m, tolerance, solver = True, repetitions = 10):

    solutions = []
    s = np.random.randint(2, size=n)

    circuit = make_circuit(s, n, not_time, swap_time, m, tolerance)

    jobid = send_to_ibm(circuit, repetitions)



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
from pprint  import pprint

'''
Parameters
'''
n = 2
not_time = 0 # Define the complexity of the Uf
swap_time = 0 # Define the complexity of the Uf
m = 100  # Trail time
tolerance = 1e-10 # Smaller the value, more accurate each trial


apikey = load_credential()
IBMQ.save_account(apikey)
make_a_run_ibm(n, not_time, swap_time, m, tolerance, repetitions=1)


assert(0)

#1. First Verify our Solution is correct

print("\n\n############ Verifying the correcness ############")
n = 6
not_time = 5 # Define the complexity of the Uf
swap_time = 5 # Define the complexity of the Uf
m = 100  # Trail time
tolerance = 1e-10 # Smaller the value, more accurate each trial

s, solution, simulation_time, linear_solver_time, circuit = make_a_run(n, not_time, swap_time, m, tolerance)
print("Random Generated s:", s)
print("Found Solution :", solution[0])
print("Circuit:")
print(circuit)

#2. Excution time with different U_f
print("\n\n############ Testing execution time of different f ############")
import matplotlib.pyplot as plt

n = 6
m = 50  # Trail time
tolerance = 1e-10 # Smaller the value, more accurate each trial
range_op = 26

simulation_time_uf = []
for i in range(1, range_op):
    s, solution, simulation_time, linear_solver_time, circuit = make_a_run(n, i, i, m, tolerance, solver = False)
    simulation_time_uf.append(simulation_time)
plt.figure()
plt.plot([i for i in range(1, range_op)], simulation_time_uf)
plt.xlabel('Complexity of Uf')
plt.ylabel('Simulation Time')
plt.savefig('Uf.png')

#3. Excution time with different n
print("\n\n############ Testing execution time of different n ############")
m = 1  # Trail time
not_time = 0 # Define the complexity of the Uf
swap_time = 0 # Define the complexity of the Uf
tolerance = 1e-10 # Smaller the value, more accurate each trial
range_n = 15

simulation_time_n = []
for i in range(1, range_n):
    print(i)
    s, solution, simulation_time, linear_solver_time, circuit = make_a_run(i, not_time, swap_time, m, tolerance, solver = False)
    simulation_time_n.append(simulation_time)
plt.figure()
plt.plot([i for i in range(1, range_n)], [math.log(i) for i in simulation_time_n])
plt.xlabel('N')
plt.ylabel('Simulation Time (Log)')
plt.savefig('n.png')