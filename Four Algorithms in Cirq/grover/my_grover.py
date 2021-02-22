from typing import List

import cirq
import numpy as np
from matplotlib import pyplot as plt
import argparse
import time
import random
import statistics
import sys

import timeout_decorator


class CustumFunction:
    def __init__(self, n, needle):
        # n is the number of bits that the function takes as input
        # needle is the x such that f(x)=1
        self.n = n
        self.needle = needle

    def __call__(self, x):
        return int(x == self.needle)

    def peek_needle(self):
        return self.needle

    def get_num_bits(self):
        return self.n


class OracleGate(cirq.Gate):
    def __init__(self, n, Z, name):
        super(OracleGate, self)
        self.n = n
        self.Z = Z
        self.name = name

    def _num_qubits_(self):
        return self.n

    def _unitary_(self):
        return self.Z

    def __str__(self):
        return self.name

    # def _circuit_diagram_info_(self, args):
    #     return self.name


def make_oracle(n, f):
    # produces Z_f, in np.array
    diag = [(-1) ** f(i) for i in range(2 ** n)]
    Z_f = np.diag(diag)
    return Z_f


def make_grover_curcuit(n, Z_f):
    Z_0 = np.eye(2 ** n)
    Z_0[0, 0] = -1
    qubits = cirq.LineQubit.range(n)

    Z_0_gate = OracleGate(n, Z_0, 'Z_0')
    Z_f_gate = OracleGate(n, Z_f, 'Z_f')

    ops = [cirq.H(q) for q in qubits] + [Z_f_gate.on(*qubits)] + [cirq.H(q) for q in qubits] + [
        Z_0_gate.on(*qubits)] + [cirq.H(q) for q in
                                 qubits] + [cirq.measure(*qubits, key='result')]
    grover_circuit = cirq.Circuit(ops)
    return grover_circuit
    print(grover_circuit)


def run_benchmark():
    print('We will do two benchmarks:')
    print('(1) We will study how different choices of f affects the runtime of Grover algorithm')
    print('(2) We will test how Grover simulation performs as the number of bits, n, increases')
    num_of_f = 1000  # will generate num_of_f different f
    num_runs_for_f = 5  # for each f, will run num_runs_for_f to reduce variation
    n = 12  # number of bits that f takes
    print('First, I will be testing how different choices of Z_f will affect the execution time by generating {} '
          'random '
          'f with n={} bits'.format(num_of_f, n))
    run_times_log_different_f: List[int] = list()

    for nof in range(num_of_f):
        print('Running Grover on {} out of {} random functions'.format(nof + 1, num_of_f))
        needle = random.randint(0, (2 ** n) - 1)
        f = CustumFunction(n, needle)
        avg_runtime_f = 0
        for run in range(num_runs_for_f):
            f = CustumFunction(n, needle)
            start = time.time()
            Z_f = make_oracle(n, f)
            grover_f = make_grover_curcuit(n, Z_f)
            simulator = cirq.Simulator()
            result = simulator.run(grover_f)
            end = time.time()
            elasped_time = end - start
            avg_runtime_f += elasped_time
        avg_runtime_f /= num_runs_for_f
        run_times_log_different_f.append(avg_runtime_f)
    print('Finished testing how different choices of Z_f will affect the execution time by generating {} random '
          'f with n={} bits'.format(num_of_f, n))
    print('Here are some statistics: ', end='')
    print('mean running time: {}, standard deviation of running time: {}'.format(
        statistics.mean(run_times_log_different_f), statistics.pstdev(run_times_log_different_f)))



    @timeout_decorator.timeout(300, timeout_exception=StopIteration)
    def timeout_wrapper(f, n):
        Z_f = make_oracle(n, f)
        grover_f = make_grover_curcuit(n, Z_f)
        simulator = cirq.Simulator()
        result = simulator.run(grover_f)

    max_n = 14
    print('Second, we will vary n from 1 to {} to get a sense of how the number of bits affect the runtime'.format(max_n))
    run_times_log_different_n = list()
    ns = list()
    timeout_occurred = 0
    for n in range(1, max_n + 1):
        needle = random.randint(0, (2 ** n) - 1)
        f = CustumFunction(n, needle)
        start = time.time()
        try:
            print('Testing runtime for n={}'.format(n))
            timeout_wrapper(f, n)
            end = time.time()
        except StopIteration:
            timeout_occurred = n
        if timeout_occurred:
            print('Experienced timeout for n={}. Will not test bigger values of n...'.format(timeout_occurred))
            break
        else:
            end = time.time()
            elasped_time = end - start
            run_times_log_different_n.append(elasped_time)
            ns.append(n)

    print('Finished testing how different values of n affect the runtime. Generating the report...')
    if timeout_occurred:
        print('Beware that we experienced a timeout (5 min) at n={}. The report below will not show any n>={}'.format(timeout_occurred, timeout_occurred))

    plt.bar(ns, run_times_log_different_n, align='center')  # A bar chart
    plt.xlabel('Number of bits')
    plt.ylabel('Grover runtimes (seconds)')
    plt.show()




def run_custom_input(n, needle):
    print('Will be running Grover algorithm on your input f(x) that takes {} bits with f({})=1'.format(n, needle))
    if (needle > (2 ** n) - 1) or (needle < 0):
        raise ValueError("Your needle cannot be represented using {} bits".format(n))
    f = CustumFunction(n, needle)
    Z_f = make_oracle(n, f)
    grover_f = make_grover_curcuit(n, Z_f)
    simulator = cirq.Simulator()
    result = simulator.run(grover_f, repetitions=10000)
    print('Now, I am going to run the Grover circuit 10000 times to search for the needle...')
    print('Measurement results')
    print(result.histogram(key='result'))
    print('If Grover circuit did it\'s job, then the most occurring key in the above result should be the needle {}'.format(needle))


if __name__ == '__main__':

    argparser = argparse.ArgumentParser(description='Demonstration of the Grover algorithm')
    argparser.add_argument('-b', '--benchmark', help='Run various benchmark', action='store_true', default=True)
    argparser.add_argument('-c', '--custom_function', action='store_true', help='Run Grover algorithm with custom input function',
                           default=False)
    argparser.add_argument('-x', '--needle',
                           help='If --custom_function is set, set the x such that f(x)=1. This should be in decimal '
                                'format')
    argparser.add_argument('-n', '--num_bits', help='If --custom_function is set, set the number of bits that f(x) is '
                                                    'expecting')

    args = vars(argparser.parse_args())

    if args['custom_function']:
        n = int(args['num_bits'])
        needle = int(args['needle'])
        run_custom_input(n, needle)
    elif args['benchmark']:
        run_benchmark()
    else:
        print('Error: You need to either provide --benchmark option or --custom_function option. Exiting',
              file=sys.stderr)
        exit(1)
