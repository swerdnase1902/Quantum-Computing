import argparse
import matplotlib.pyplot as plt

import random
import cirq
from cirq import H, X, CNOT, measure


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

    args = vars(argparser.parse_args())

    
    if args['benchmark']:
        time_mean = []
        time_std = []
        # measure the average runtime over several trials
        repeat = 100
        # length of input strings
        for n in range(2, 17):            
            ts = []           
            for _ in range(repeat):  
                t1 = time.perf_counter()
                run_algo(n, visual = False)
                t2 = time.perf_counter()
                ts.append(t2 - t1)
            # print(t2 - t1)
            time_mean.append(np.mean(ts))
            time_std.append(np.std(ts))
        plot_time(time_mean, time_std)        
    elif args['num_bits']:        
        t1 = time.perf_counter()
        n = int(args['num_bits'])
        run_algo(n, visual = True)
        t2 = time.perf_counter()
        print(f"finish in {t2 - t1} seconds")
    else:
        print(
            'Error: You need to either provide --benchmark option', 
            'or --num_bits option. Exiting', file=sys.stderr
        )
        exit(1)