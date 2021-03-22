"""
Implementation of Shor's algorithm
References:
[1] https://qiskit.org/textbook/ch-algorithms/shor.html
[2] https://quantumai.google/cirq/tutorials/shor
"""
import argparse
import fractions
from fractions import Fraction
import math
from math import gcd
import random
from typing import Callable, List, Optional, Sequence, Union

import sympy

import cirq
from cirq import H, X, CNOT, measure, qft, SWAP
import requests

def c_amod15(a, power, qbts, expq):
    if a not in [2,7,8,11,13]:
        raise ValueError("'a' must be 2,7,8,11 or 13")
    assert len(qbts) == 4
    U = []
    for iteration in range(power):        
        U.append(SWAP(qbts[2],qbts[3]).controlled_by(expq))
        U.append(SWAP(qbts[1],qbts[2]).controlled_by(expq))
        U.append(SWAP(qbts[0],qbts[1]).controlled_by(expq))
        for q in qbts:
            U.append(X(q).controlled_by(expq))    
    return U

def make_order_finding_circuit(x: int, n: int) -> cirq.Circuit:
    n_count = 8
    L = n.bit_length()
    target = cirq.LineQubit.range(L)
    exponent = cirq.LineQubit.range(L, 3 * L)
    print(f"x = {x}, n = {n}, L = {L}, 3 * L = {3 * L}")
    c = cirq.Circuit()
    c.append([X(target[L - 1])])
    c.append([H.on_each(*exponent)])
    # ModularExp(target, exponent, x, n),
    for q in range(n_count):
        c.append(c_amod15(x, 2**q, target, exponent[q]))
    c.append([qft(*exponent, inverse=True)])
    c.append([measure(*exponent, key='exponent')])
    return c


def read_eigenphase(result: cirq.Result) -> float:    
    exponent_as_integer = result.data['exponent'][0]    
    exponent_num_bits = result.measurements['exponent'].shape[1]
    return float(exponent_as_integer / 2 ** exponent_num_bits)


def load_credential():
    return "zhezeng@cs.ucla.edu", 605243830


def quantum_order_finder(x, n, simulation=True):    
    if x < 2 or n <= x or math.gcd(x, n) > 1:
        raise ValueError(f'Invalid x={x} for modulus n={n}.')

    circuit = make_order_finding_circuit(x, n)
    if simulation:
        simulator = cirq.Simulator()
        result = simulator.run(circuit)   
        print(result)
        print(f"type of result = {type(result)}")
        # result = cirq.sample(circuit)
    else:
        url = 'http://quant-edu-scalability-tools.wl.r.appspot.com/send'
        email, uid = load_credential()
        job_payload = {
            "circuit":cirq.to_json(circuit), 
            "email":email, 
            "repetitions":1,
            "student_id":uid
        }
        response = requests.post(url, json=job_payload)
        print(response.text)
        exit()
    phase = read_eigenphase(result)
    return phase


def find_factor_of_prime_power(n: int) -> Optional[int]:
    """Returns non-trivial factor of n if n is a prime power, else None."""
    for k in range(2, math.floor(math.log2(n)) + 1):
        c = math.pow(n, 1 / k)
        c1 = math.floor(c)
        if c1 ** k == n:
            return c1
        c2 = math.ceil(c)
        if c2 ** k == n:
            return c2
    return None


def find_factor(n, order_finder, max_attempts= 30):
    x = 7        
    for _ in range(max_attempts):
        phase = quantum_order_finder(x, n, simulation=False)
        frac = Fraction(phase).limit_denominator(n) # Denominator should (hopefully!) tell us r
        r = frac.denominator
        print("Result: r = %i" % r)    
        if phase != 0:
            # Guesses for factors are gcd(x^{r/2} ±1 , 15)
            guesses = [gcd(x**(r//2)-1, n), gcd(x**(r//2)+1, n)]
            print("Guessed Factors: %i and %i" % (guesses[0], guesses[1]))
            for guess in guesses:
                if guess not in [1,n] and (n % guess) == 0: # Check to see if guess is a factor
                    print("Non-trivial factor found: %i" % guess)                
                    return guess
    return None


def main(n):
    if n < 2:
        raise ValueError(f'Invalid input {n}, expected positive integer greater than one.')

    d = find_factor(n, quantum_order_finder)

    if d is None:
        print(f'No non-trivial factor of {n} found. It is probably a prime.')
    else:
        print(f'{d} is a non-trivial factor of {n}')

        assert 1 < d < n
        assert n % d == 0


if __name__ == '__main__':    
    parser = argparse.ArgumentParser(description='Factorization demo.')
    parser.add_argument('n', type=int, help='composite integer to factor')
    args = parser.parse_args()
    main(n=args.n)