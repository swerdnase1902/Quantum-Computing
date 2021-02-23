import argparse
import matplotlib.pyplot as plt

import random
import cirq
from cirq import H, X, CNOT, measure


def make_oracle(qbts, n, constant = -1, visual = False):
    """
    Input:    
    qbts -- qubits in the circuit;
    n -- length of the input bit string;
    constant -- 1 for generate a constant f, 0 for generate a balanced f, 
        -1 for f being constant or balanced at random;
    Output:
    oracle -- gate implementation of f
    """
    if constant < 0:
        constant = random.randint(0, 1)
    
    if constant:  # f being constant
        if visual:
            print("generate constant f")
        if random.randint(0, 1):  # constant output of f
            return [X(qbts[-1])]
        else:
            return []
    else:
        if visual:
            print("generate balanced f")
        gates = []
        for qbt in qbts[:-1]:
            if random.randint(0, 1):  # randomly flip qubits            
                gates.extend([X(qbt), CNOT(qbt, qbts[-1]), X(qbt)])
            else:
                gates.append(CNOT(qbt, qbts[-1]))
        return gates


def make_circuit(qbts, oracle):
    """
    Input:
    qbts -- qubits in the circuit;
    oracle -- the circuit implementation of the black-box function f;
    Output:
    c -- circuit that implements the Deutsch-Jozsa algorithm
    """
    c = cirq.Circuit()    
    c.append([X(qbts[-1])])
    c.append([H(qbt) for qbt in qbts])
    # Initialize qubits.
    # c.append([X(q1), H(q1), H(q0)])

    # Query oracle.
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
    
    oracle = make_oracle(qbts, n, constant, visual = visual)
    circuit = make_circuit(qbts, oracle)
    if visual:
        print('Circuit:')
        print(circuit)

    # Simulate the circuit.
    simulator = cirq.Simulator()
    result = simulator.run(circuit)    
    # if measured to be (00...0) then f is constant else balanced
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
    plt.savefig("dj_time_mean.png")

    plt.clf()
    # logy = [np.log(t) for t in time_std]
    logy = time_std
    plt.plot(ns, logy, "o-")
    plt.xlabel("input bit string length")
    plt.ylabel("standard deviation of runtime in log (seconds)")    
    plt.savefig("dj_time_std.png")


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
        '-t', '--type', 
        help='specify the type of generated function f, '
        '1 for constant, 0 for balanced, -1 for random', default=-1
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
            print(t2 - t1)
            time_mean.append(np.mean(ts))
            time_std.append(np.std(ts))
        plot_time(time_mean, time_std)        
    elif args['num_bits']:
        constant = int(args['type'])
        t1 = time.perf_counter()
        n = int(args['num_bits'])
        run_algo(n, visual = True, constant = constant)
        t2 = time.perf_counter()
        print(f"finish in {t2 - t1} seconds")
    else:
        print(
            'Error: You need to either provide --benchmark option', 
            'or --num_bits option. Exiting', file=sys.stderr
        )
        exit(1)
    

