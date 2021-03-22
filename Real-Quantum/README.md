
Chiao Lu: UID
Harold: UID
Zhe Zeng: UID

# Describe your approach to error correction.  Compare runs with and without error correction.
## BV
## DJ
## Grover
## Simon
Following the approach in https://en.wikipedia.org/wiki/Quantum_error_correction#/(Bit flip code), we use two additional help qubits per input qubit to correct bit flip code. 

On local simulation, we find that the error error correction is correct in that it will not incorrectly change the original correct result. We test with two input qubits and the random generated string is 01. Both simulations with/without error correction could find the correct answer.

On IBM's computer, we find that the real quantum computers are still error-prone. Without error correction, the results from the quantum computer do not produce a valid answer (the linear solver fails with the returned parameters); with erro correction, the result we found is 10, which is wrong.

Google's computer does not support CCNOT so we cannot run error correction.

## QAOA
Joseph
## Shor

# Discuss your effort to test the programs and present results from the testing. Run each program multiple times and present statistics of the results.
## BV
## DJ
## Grover
## Simon
We are able to run on IBM and below is the result we got from IBM. It seems the computer makes a lot of errors. ![simon_ibm](figures/simon_ibm.png)

On Google, the computer still makes errors. Out of 10000 runs, only 2053 of them gave the correct answer. The most frequent results produce the wrong answer 10 (the correct string is 01).
## QAOA
Joseph
## Shor

# What is your experience with scalability as n grows?  Present a diagram that maps n to execution time.

Not enough information for this section. As Google and IBM don't let us test this.

# Compare your results across the two quantum computers
## BV
## DJ
## Grover
## Simon
On IBM, the results are all wrong with or without error correction.

On Google, the results seem almost random and is still wrong.

## QAOA
On IBM, we got pretty good results. It can be seen that the state 11 gets the most hits, which is the correct solution. ![qaoa_ibm](figures/qaoa_ibm.png)

Google does not support CCNOT gate so we are not able to get QAOA on Google.
## Shor

# Compare your results from running Cirq programs on the Google quantum computer with your results from running those same Cirq programs on the simulator.
## BV
## DJ
## Grover
## Simon
The results from Google is still very unstable. On simulators, we can get the correct answer on very few trys but on Google, the results do not seem to improve even if we increase the number of runs, possbily due to very high error rates.
## QAOA
Joseph
## Shor

# how to provide input, how to run the program, and how to understand the output
## BV
## DJ
## Grover
## Simon
To test on IBM, simply run "python qiskit/simon.py" and the program will ran an example. It will first run a simulation and then connect IBM to run the quantmn computer. It will run two versions: simulated and IBM.

To test on Google, simply run "python qiskit/QAOA.py" and the program will ran an example. It will first run a simulation and then connect Google to run the quantmn computer. After we get the job id and the run finishes on Goolge, we could call check_result_with_ids() to solve the retured results.

## QAOA
To test on IBM, simply run "python qiskit/QAOA.py" and the program will ran an example. It will first run a simulation and then connect IBM to run the quantmn computer. It will run four versions: 1) simulated, 2) simulated-error-correction, 3) IBM, and 4) IBM-error-correction.

## Shor

# Which modifications of the programs did you have to make when moving from the simulator to the quantum computer?
Work on this together. Make some bullet points first
### Remove custom gates
In some of the simulation codes, we used custom gates that are defined by unitary matrixes. However, when moving to quantum computers, we need to decompose them into basic gates. This is not an easy task as the general decomposition of a custom gate into basic gates is a hard problem.

To circumvent the issue, we manually designed specific examples and decomposed the custom gates by hand.

### Error correction
When running on simulators, there is no error. But when moving to quantum computers, errors become frequent and it is hard to get a reliable result. We implemented error correction to our best effort, but the current quantum computers can still produce many errors.

### Querying from the server
When running on Goolge, we have to check the results the next day, which is inconvenient. When running on IBM, the queue could very long and sometimes we need to wait for several hours for our program to be run.
