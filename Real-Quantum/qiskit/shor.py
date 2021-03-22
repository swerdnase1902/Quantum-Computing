"""
Implementation of Shor's algorithm
References:
[1] https://qiskit.org/textbook/ch-algorithms/shor.html
[2] https://quantumai.google/cirq/tutorials/shor
"""
import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit, Aer, transpile, assemble, IBMQ
from qiskit.visualization import plot_histogram
from math import gcd
from numpy.random import randint
import pandas as pd
from fractions import Fraction
from math import gcd


def c_amod15(a, power):
    """Controlled multiplication by a mod 15"""
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    U = QuantumCircuit(4)        
    for iteration in range(power):
        if a in [2,13]:
            U.swap(0,1)
            U.swap(1,2)
            U.swap(2,3)
        if a in [7,8]:
            U.swap(2,3)
            U.swap(1,2)
            U.swap(0,1)
        if a == 11:
            U.swap(1,3)
            U.swap(0,2)
        if a in [7,11,13]:
            for q in range(4):
                U.x(q)
    U = U.to_gate()
    U.name = "%i^%i mod 15" % (a, power)
    c_U = U.control()
    return c_U

def qft_dagger(n):
    """n-qubit QFTdagger the first n qubits in circ"""
    qc = QuantumCircuit(n)
    # Don't forget the Swaps!
    for qubit in range(n//2):
        qc.swap(qubit, n-qubit-1)
    for j in range(n):
        for m in range(j):
            qc.cp(-np.pi/float(2**(j-m)), m, j)
        qc.h(j)
    qc.name = "QFT†"
    return qc

def load_credential():
    return "4471a7286c3a12f0749ed152dba35e8bb6237d4802007e1023bcce4dc89358edc40ebb8696fc608ae1c9872cd41d6ad4aeec867e9a186cafa929816cf2aa109e"

def qpe_amod15(a, simulation=True):
    n_count = 8
    qc = QuantumCircuit(4+n_count, n_count)
    for q in range(n_count):
        qc.h(q)     # Initialise counting qubits in state |+>
    qc.x(3+n_count) # And auxiliary register in state |1>
    for q in range(n_count): # Do controlled-U operations
        qc.append(c_amod15(a, 2**q), 
                 [q] + [i+n_count for i in range(4)])
    qc.append(qft_dagger(n_count), range(n_count)) # Do inverse-QFT
    qc.measure(range(n_count), range(n_count))
    
    if simulation:
        # Simulate Results
        qasm_sim = Aer.get_backend('qasm_simulator')
        # Setting memory=True below allows us to see a list of each sequential reading
        t_qc = transpile(qc, qasm_sim)
        qobj = assemble(t_qc, shots=1)
        result = qasm_sim.run(qobj, memory=True).result()
        readings = result.get_memory()
        print("Register Reading: " + readings[0])
        phase = int(readings[0],2)/(2**n_count)
        print("Corresponding Phase: %f" % phase)
    else:
        apikey = load_credential()
        IBMQ.save_account(apikey)
        num_shots = 1
        provider = IBMQ.load_account()        
        backend = provider.backends.ibmq_16_melbourne

        transpiled = transpile(qc, backend)
        qobj = assemble(transpiled, backend, shots=num_shots)
        job = backend.run(qobj)
        print(job.job_id())
        result = job.result()
        counts = result.get_counts()
        delayed_result = backend.retrieve_job(job.job_id()).result()
        delayed_counts = delayed_result.get_counts()
        print(counts)
        readings = list(counts.keys())
        phase = int(readings[0],2)/(2**n_count)
    return phase


N = 15
a = 7
factor_found = False
attempt = 0
while not factor_found:
    attempt += 1
    print("\nAttempt %i:" % attempt)
    phase = qpe_amod15(a, simulation=False) # Phase = s/r
    frac = Fraction(phase).limit_denominator(N) # Denominator should (hopefully!) tell us r
    r = frac.denominator
    print("Result: r = %i" % r)
    if phase != 0:
        # Guesses for factors are gcd(x^{r/2} ±1 , 15)
        guesses = [gcd(a**(r//2)-1, N), gcd(a**(r//2)+1, N)]
        print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
        for guess in guesses:
            if guess not in [1,N] and (N % guess) == 0: # Check to see if guess is a factor
                print("Non-trivial factor found: %i" % guess)
                factor_found = True
