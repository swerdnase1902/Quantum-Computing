"""
Implementation of Shor's algorithm
References:
[1] https://qiskit.org/textbook/ch-algorithms/shor.html
[2] https://quantumai.google/cirq/tutorials/shor
"""

from sympy import isprime
import numpy as np
from numpy import random
import math
import fractions
import matplotlib.pyplot as plt

import time
import cirq


def primepower(n):
    """    
    Input:
    n -- an integer
    Output:
    if n is the power of a prime i, then return i; else return None.
    """
    for i in range(2, int(np.sqrt(n) + 1)):
        if not isprime(i) or not (n % i == 0):
            continue        
        while n % i == 0:
            n //= i            
        if n == 1:  # n is the power of prime i
            return i
        else:
            return None        


class MyModular(cirq.ArithmeticOperation):    
    def __init__(self, target, exponent, base, modulus):        
        self.target = target
        self.exponent = exponent
        self.base = base
        self.modulus = modulus

    def apply(self, *register_values):
        target, exponent, base, modulus = register_values        
        if target >= modulus:
            return target
        return (target * base ** exponent) % modulus

    def registers(self):
        return self.target, self.exponent, self.base, self.modulus
    
    def with_registers(self, *new_registers):
        target, exponent, base, modulus = new_registers
        return MyModular(target, exponent, base, modulus)


def make_circuit(a, n):
    """
    Input:
    a -- integer in range (1, n)    
    n -- integer
    Output:
    c -- the quantum circuit for Shor's algorithm
    """
    m = n.bit_length()
    # choose qubits to use        
    qbts1 = cirq.LineQubit.range(m)
    qbts2 = cirq.LineQubit.range(m, 3 * m + 3)
    # create circuit
    c = cirq.Circuit()    
    c.append(cirq.X(qbts1[m - 1]))
    c.append(cirq.H.on_each(*qbts2))
    c.append(MyModular(qbts1, qbts2, a, n))    
    c.append(cirq.qft(*qbts2, inverse=True))
    c.append(cirq.measure(*qbts2, key='results'))
    return c


def find_order(a, n):
    """
    Input:
    a -- integer in range (1, n)    
    n -- integer
    Output:
    r -- the smallest integer such that a ** r % n = 1
    """
    circuit = make_circuit(a, n)
    res = cirq.sample(circuit)
    # print(f"results from circuit: {res.data['results'][0]}")
    phase = float(res.data['results'][0] / 2 ** res.measurements['results'].shape[1])
    f = fractions.Fraction.from_float(phase).limit_denominator(n)  # fractions
    if f.numerator == 0:
        return None
    r = f.denominator
    if a ** r % n == 1:
        return r
    return None


def unit_test():
    """
    tests for correctness of the helper functions
    """
    assert primepower(4) == 2
    assert primepower(27) == 3    
    assert primepower(24) is None
    assert primepower(36) is None
    print("Pass all the tests.")


def run(n, n_trials=10, benchmark=False):
    """
    Input:
    n -- the integer to factorize
    n_trials -- the repeat times in the Shor's algorithm
    benchmark -- whether running the benchmark or not
    Output:
    d -- a non-trivial factor of integer n
    or save the benchmark results
    """        
    assert n > 1, f"the input integer should be greater than 1"
    if n % 2 == 0:
        return 2
    if isprime(n):  # n is a prime
        return n
    d = primepower(n)
    if d:  # n is the power of a prime
        return d
    if not benchmark:
        for _ in range(n_trials):
            a = random.randint(2, n - 1)
            d = math.gcd(a, n)            
            if 1 < d < n:
                return d  # d is a non-trivial factor of n
            r = find_order(a, n)
            if r is None or r % 2 != 0:
                continue
            x = (a ** (r // 2) - 1) % n
            d = math.gcd(x, n)
            if d > 1:
                return d
        
    else: 
        # benchmark over integer x in [2, n - 1]
        time_mean = []
        time_std = []
        for x in range(2, n):
            ts = []
            for _ in range(n_trials):                
                t1 = time.perf_counter()
                c = math.gcd(x, n)
                if 1 < c < n:
                    t2 = time.perf_counter()
                    ts.append(t2 - t1)
                    continue
                r = find_order(x, n)
                if r is None:
                    t2 = time.perf_counter()
                    ts.append(t2 - t1)                    
                    continue
                if r % 2 != 0:
                    t2 = time.perf_counter()
                    ts.append(t2 - t1)                    
                    continue
                y = x ** (r // 2) % n
                assert 1 < y < n
                c = math.gcd(y - 1, n)
                if 1 < c < n:
                    t2 = time.perf_counter()
                    ts.append(t2 - t1)                    
                    continue                
                t2 = time.perf_counter()
                ts.append(t2 - t1)            
            time_mean.append(np.mean(ts))
            time_std.append(np.std(ts))
        plot_time(time_mean, time_std, tag=f"shor_{n}")
    return None


def plot_time(time_mean, time_std, tag="shor"):
    n = len(time_mean)
    fig = plt.figure()    
    ns = [f'{2 + i}' for i in range(n)]
    logy = [np.log(t) for t in time_mean]
    plt.plot(ns, logy, "o-")
    plt.xlabel("input bit string length")
    plt.ylabel("avg. runtime in log (seconds)")    
    plt.savefig(f"{tag}_time_mean.png")

    plt.clf()
    # logy = [np.log(t) for t in time_std]
    logy = time_std
    plt.plot(ns, logy, "o-")
    plt.xlabel("input bit string length")
    plt.ylabel("standard deviation of runtime (seconds)")
    plt.savefig(f"{tag}_time_std.png")


if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser(description='shor algorithm')
    argparser.add_argument('-m', '--max', type=int, help='max number of trials')    
    argparser.add_argument(
        '-b', '--benchmark', help='Run various benchmark', 
        action='store_true', default=False
    )
    argparser.add_argument('n', type=int, help='integer to factor')

    args = vars(argparser.parse_args())
    if args['benchmark']:        
        unit_test()  # tests for helper functions
        time_mean = []
        time_std = []
        n = args['n']
        repeat = 10        
        for i in range(3, n + 1):  # benchmark from 3 to n
            ts = []
            for _ in range(repeat):
                t1 = time.perf_counter()
                d = run(i, n_trials=args['max'], benchmark=False)
                # print(f"return result = {d}")
                t2 = time.perf_counter()
                ts.append(t2 - t1)
            time_mean.append(np.mean(ts))
            time_std.append(np.std(ts))
        plot_time(time_mean, time_std)
        print(f"time mean: {time_mean}")
        print(f"time std : {time_std}")
        
        for i in [15, 21]:
            if i > n:
                continue
            run(i, n_trials=args['max'], benchmark=True)

    else:
        t1 = time.perf_counter()
        d = run(n=args['n'], n_trials=args['max'], benchmark=False)
        t2 = time.perf_counter()
        print(f"finish in {t2 - t1} seconds")
        if d:
            print(f"found factor {d} for input integer {args['n']}")
        else:
            print(f"factor not found")