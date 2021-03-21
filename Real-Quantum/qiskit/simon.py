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
import numpy
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
def oracle(inputs, outputs, s, circuit, n, swap_time = 0, not_time = 0, other_inputs = None): 
    # 1
    for index in range(len(inputs)):
        circuit.append(cirq.CNOT(inputs[index], outputs[index]))  # Becuase outputs_quibuts are all zeros, we can do the copy with CNOT
        if other_inputs[0] is not None:
            circuit.append(cirq.CNOT(other_inputs[0][index], outputs[index]))
            circuit.append(cirq.CNOT(other_inputs[1][index], outputs[index]))

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
                if other_inputs[0] is not None:
                    circuit.append(cirq.CNOT(other_inputs[0][chosen_index], outputs[i]))
                    circuit.append(cirq.CNOT(other_inputs[1][chosen_index], outputs[i]))
            
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


def send_to_ibm(circuit: cirq.Circuit, repetitions):
    circ = circuit
    qbits = circ.all_qubits()

    provider = IBMQ.load_account()
    # backend = provider.backends.backend_name
    backend = provider.backends.ibmq_16_melbourne

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
    #delayed_result = backend.retrieve_job(job.job_id()).result()
    #delayed_counts = delayed_result.get_counts()
    return counts

def make_circuit(s, n, not_time, swap_time, m, tolerance, error_correction = False):

    # Initializing the qubits
    inputs = [cirq.GridQubit(i, 0) for i in range(n)]  # inputs x
    outputs = [cirq.GridQubit(i + n, 0) for i in range(n)]
    if error_correction:
        inputs_1 = [cirq.GridQubit(i + 2 * n, 0) for i in range(n)]
        inputs_2 = [cirq.GridQubit(i + 3 * n, 0) for i in range(n)]
    else:
        inputs_1 = None
        inputs_2 = None
    circuit = cirq.Circuit()

    if error_correction:
        for i in range(n):
            circuit.append(cirq.CNOT(inputs[i], inputs_1[i]))
            circuit.append(cirq.CNOT(inputs[i], inputs_2[i]))
        

    # 1. Apply H^N to the input quibuts
    for i in range(n):
        circuit.append(cirq.H(inputs[i]))
    
        if error_correction:
            circuit.append(cirq.H(inputs_1[i]))
            circuit.append(cirq.H(inputs_2[i]))

    # 2. Add U_f
    oracle(inputs, outputs, s, circuit, n, swap_time=swap_time, not_time=not_time, other_inputs = (inputs_1, inputs_2))

    # 3. Apply H^N to the inputs qubits
    for i in range(n):
        circuit.append(cirq.H(inputs[i]))
        if error_correction:
            circuit.append(cirq.H(inputs_1[i]))
            circuit.append(cirq.H(inputs_2[i]))
    
    if error_correction:
        for i in range(n):
            circuit.append(cirq.CNOT(inputs[i], inputs_1[i]))
            circuit.append(cirq.CNOT(inputs[i], inputs_2[i]))
            circuit.append(cirq.TOFFOLI(inputs_2[i], inputs_1[i], inputs[i]))

    # 4. Measurement
    circuit.append(cirq.measure(*inputs, key='result'))

    #print("Randomly constructed circuit:")
    #print(circuit)
    return circuit

def make_a_run_ibm(s, n, not_time, swap_time, m, tolerance, solver = True, error_correction = False, repetitions = 10):

    solutions = []

    circuit = make_circuit(s, n, not_time, swap_time, m, tolerance, error_correction)

    delayed_counts = send_to_ibm(circuit ,repetitions=repetitions)
    return delayed_counts


def make_a_run(s, n, not_time, swap_time, m, tolerance, solver = True, error_correction = False):

    solutions = []

    circuit = make_circuit(s, n, not_time, swap_time, m, tolerance, error_correction)

    #print("Randomly constructed circuit:")
    #print(circuit)

    '''
    Start trial
    '''

    simulation_time = 0.0
    linear_solver_time = 0.0

    for _ in range(m):
        start_time = time.time()

        equations = [cirq.Simulator().run(circuit).measurements['result'][0] for _ in range(n - 1)]  # Get n equations
        
        simulation_time += time.time() - start_time

        if solver:
            start_time = time.time()
            # Solve the n equations given a matrix
            flag, solution = linear_solver(equations, tolerance)
            linear_solver_time += time.time() - start_time
            if flag:
                solutions.append(str(solution))

    freqs = Counter(solutions)
    #print('Found solution: {}'.format(freqs.most_common(1)[0]))
    if solver:
        return s, freqs.most_common(1)[0], simulation_time / m, linear_solver_time / m, circuit
    else:
        return s, None, simulation_time / m, linear_solver_time / m, circuit

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def solve(results, n):
    new_results = []
    for key, value in results.items():
        qubit = np.array([int(i) for i in key])
        print(qubit.shape)
        for i in range(value):
            new_results.append(qubit)
    results = list(chunks(new_results, n - 1))
    solutions = []
    for i in range(len(results)):
        equations = results[i]
        flag, solution = linear_solver(equations, tolerance=1e-6)
        if flag:
                solutions.append(str(solution))
    freqs = Counter(solutions)
    try:
        return freqs.most_common(1)[0]
    except:
        return [None]

'''from pprint  import pprint
data = lookup_google()
pprint(list(data["Jobs by student_id"]["005271406"]["5509413305581568"].keys()))
pprint(data["Jobs by student_id"]["005271406"]["5509413305581568"]["message"])
print(type(data))
print()
assert(0)'''

'''
Parameters
'''
n = 2
not_time = 0 # Define the complexity of the Uf
swap_time = 0 # Define the complexity of the Uf
m = 100  # Trail time
tolerance = 1e-10 # Smaller the value, more accurate each trial

print(solve({'00': 1}, n))
#assert(0)


random.seed(0)
numpy.random.seed(0)
s = np.random.randint(2, size=n)
print("Generated Random s: ", s)

#1. Simulation
s, solution, simulation_time, linear_solver_time, circuit = make_a_run(s, n, not_time, swap_time, m, tolerance)
print("Found Solution (Simulated):", solution[0])

#2. Simulation with error correction
s, solution, simulation_time, linear_solver_time, circuit = make_a_run(s, n, not_time, swap_time, m, tolerance, error_correction=True)
print("Found Solution (Simulated, error correction):", solution[0])

# 3. Without error correction
results = make_a_run_ibm(s, n, not_time, swap_time, m, tolerance, repetitions=50 * (n-1))
solution = solve(results, n)
print("Found Solution (IBM):", solution[0])

# 4. With error correction
results = make_a_run_ibm(s, n, not_time, swap_time, m, tolerance, error_correction = True, repetitions=50 * (n-1))
solution = solve(results, n)
print("Found Solution (IBM, error correction):", solution[0])
