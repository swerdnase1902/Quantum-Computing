import argparse
import matplotlib.pyplot as plt

import random
import cirq
from cirq import H, X, CNOT, measure
import requests

import qiskit
from qiskit import IBMQ, assemble, transpile


def make_oracle(qbts, n, visual = False):
    """
    Input:    
    qbts -- qubits in the circuit;
    n -- length of the input bit string;    
    Output:
    oracle -- gate implementation of f
    """
    bias = random.randint(0, 1)
    coef = [random.randint(0, 1) for _ in range(n)]    
    if visual:
        print(f"coef: {coef}")
    gates = []
    if bias:
        gates.append(X(qbts[-1]))
    for i, qbt in enumerate(qbts[:-1]):
        if coef[i]:
            gates.append(CNOT(qbt, qbts[-1]))
    return gates


def make_circuit(qbts, oracle):
    """
    Input:
    qbts -- qubits in the circuit;
    oracle -- the circuit implementation of the black-box function f;
    Output:
    c -- circuit that implements the Bernstein-Vazirani algorithm
    """
    c = cirq.Circuit()    
    c.append([X(qbts[-1])])
    c.append([H(qbts[-1])])
    c.append([H(qbt) for qbt in qbts[:-1]])
    c.append(oracle)
    c.append([H(qbt) for qbt in qbts[:-1]])
    # Measure
    m = []
    for i, qbt in enumerate(qbts[:-1]):
        m.append(measure(qbt, key=f'result-{i}'))    
    c.append(m)
    return c


def run_algo(n, visual = True, constant = -1):
    # Choose qubits to use.
    qbts = cirq.LineQubit.range(n + 1)
    
    oracle = make_oracle(qbts, n, visual = visual)
    circuit = make_circuit(qbts, oracle)
    if visual:
        print('Circuit:')
        print(circuit)

    # Simulate the circuit.
    simulator = cirq.Simulator()
    result = simulator.run(circuit)    
    if visual:
        print(result)  


def plot_time(time_mean, time_std):
    n = len(time_mean)
    fig = plt.figure()    
    ns = [f'{2 + i}' for i in range(n)]
    logy = [np.log(t) for t in time_mean]
    plt.plot(ns, logy, "o-")
    plt.xlabel("input bit string length")
    plt.ylabel("avg. runtime in log (seconds)")    
    plt.savefig("bv_time_mean.png")

    plt.clf()
    # logy = [np.log(t) for t in time_std]
    logy = time_std
    plt.plot(ns, logy, "o-")
    plt.xlabel("input bit string length")
    plt.ylabel("standard deviation of runtime (seconds)")    
    plt.savefig("bv_time_std.png")

def load_credential():
    return "4471a7286c3a12f0749ed152dba35e8bb6237d4802007e1023bcce4dc89358edc40ebb8696fc608ae1c9872cd41d6ad4aeec867e9a186cafa929816cf2aa109e"


if __name__ == '__main__':
    import time
    import numpy as np   
    from qiskit.providers.ibmq import least_busy 

    argparser = argparse.ArgumentParser(
        description='Demonstration of the Deutsch-Jozsa algorithm'
    )
    argparser.add_argument(
        '-b', '--benchmark', help='Run various benchmark', 
        action='store_true', default=False
    )    
    argparser.add_argument(
        '-n', '--num_bits', 
        help='set the number of bits that f(x) is expecting'
    )    

    args = vars(argparser.parse_args())    
    n = int(args['num_bits'])

    apikey = load_credential()
    IBMQ.save_account(apikey)
    num_shots = 1
    provider = IBMQ.load_account()
    backend = provider.backend.ibmq_athens
    # backend = least_busy(provider.backends(filters=lambda x: x.configuration().n_qubits >= (n+1) and not x.configuration().simulator and x.status().operational==True))

    qbts = cirq.LineQubit.range(n + 1)
    oracle = make_oracle(qbts, n, visual = True)
    circuit = make_circuit(qbts, oracle)    
    qasm_circuit = circuit.to_qasm()
    circuit = qiskit.circuit.QuantumCircuit.from_qasm_str(qasm_circuit)

    transpiled = transpile(circuit, backend)

    qobj = assemble(transpiled, backend, shots=num_shots)
    job = backend.run(qobj)
    print(job.job_id())
    result = job.result()
    counts = result.get_counts()
    delayed_result = backend.retrieve_job(job.job_id()).result()
    delayed_counts = delayed_result.get_counts()
    print(counts)
    print(delayed_counts)
