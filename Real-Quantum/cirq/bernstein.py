import argparse
import matplotlib.pyplot as plt

import random
import cirq
from cirq import H, X, CNOT, measure, TOFFOLI
import requests


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

def make_circuit_error_correction(qbts):
    """
    hard coded for the case when n = 1 and f has a = 0 and b = 0
    Input:
    qbts -- qubits in the circuit;
    oracle -- the circuit implementation of the black-box function f;
    Output:
    c -- circuit that implements the Deutsch-Jozsa algorithm
    """        
    assert len(qbts) == 6
    c = cirq.Circuit()
    c.append([CNOT(qbts[0], qbts[2])])
    c.append([CNOT(qbts[0], qbts[4])])

    # bernstein part
    c.append([X(qbts[1]), X(qbts[3]), X(qbts[5])])
    c.append([H(qbt) for qbt in qbts])
    c.append([H(qbts[0]), H(qbts[2]), H(qbts[4])])    

    c.append([CNOT(qbts[0], qbts[2])])
    c.append([CNOT(qbts[0], qbts[4])])
    c.append([TOFFOLI(qbts[4], qbts[2], qbts[0])])

    c.append(measure(qbts[0], key='result'))
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
    return "zhezeng@cs.ucla.edu", 605243830

def lookup_google(jobid=None):
    email, uid = load_credential()
    lookup_url = 'http://quant-edu-scalability-tools.wl.r.appspot.com/lookup'

    if jobid:        
        lookup_ids = {'job_id': [jobid], 'student_id': uid}
    else:
        lookup_ids = {'student_id': uid}
    response = requests.get(lookup_url, params=lookup_ids)
    print(response)
    data = response.json()
    return data


if __name__ == '__main__':
    import time
    import numpy as np    

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
    argparser.add_argument(
        '-c', '--correction', help='Do error correction', 
        action='store_true', default=False
    )
    argparser.add_argument(
        '-s', '--simulation', help='Do simulation or run on real machine', 
        action='store_true', default=False
    )

    args = vars(argparser.parse_args())

    n = int(args['num_bits'])
    correction = bool(args['correction'])
    simulation = bool(args['simulation'])

    if not correction:
        qbts = cirq.LineQubit.range(n + 1)
        oracle = make_oracle(qbts, n, visual = True)
        circuit = make_circuit(qbts, oracle)
    else:
        print("Doing error correction")
        qbts = cirq.LineQubit.range(3 * (n + 1))
        circuit = make_circuit_error_correction(qbts)    

    if not simulation:
        url = 'http://quant-edu-scalability-tools.wl.r.appspot.com/send'
        email, uid = load_credential()
        job_payload = {
            "circuit":cirq.to_json(circuit), 
            "email":email, 
            "repetitions":50, 
            "student_id":uid
        }
        response = requests.post(url, json=job_payload)
        print(response.text)
    
    else:
        print(f"Running Bernstein with n = {n}")
        # python3 bernstein.py -n 1 -c -s
        simulator = cirq.Simulator()
        result = simulator.run(circuit)   
        print(result)