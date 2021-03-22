
Chiao Lu: UID
Harold: UID
Zhe Zeng: UID

# Describe your approach to error correction.  Compare runs with and without error correction.

Following the approach in https://en.wikipedia.org/wiki/Quantum_error_correction#/(Bit flip code), we use two additional help qubits per input qubit to correct bit flip code. 

## BV

On the local simulation, we test the error correction by randomly flipping one of the input qubit or helper qubits. It turns out that the circuit output is robust to the random flip in the sense that the flip does not change the correctness of the circuit output and it remains the same as the case when there is no random flip.

On the IBM machine, we find that even with the error correction, the quantum computers are still error-prone. We run the Bernstein-Vazirani circuit for a hard-coded case with input bit length being one for multiple trials. The results is correct for 37 trials out of a total of 50 trials.

## DJ

On the local simulation, we test the error correction by randomly flipping one of the input qubit or helper qubits. It turns out that the circuit output is robust to the random flip in the sense that the flip does not change the correctness of the circuit output and it remains the same as the case when there is no random flip.

On the IBM machine, we find that even with the error correction, the quantum computers are still error-prone. We run the Deutsch-Josza circuit for a hard-coded case with input bit length being one for multiple trials. The results is correct for 38 trials out of a total of 50 trials.

## Grover
## Simon
On local simulation, we find that the error error correction is correct in that it will not incorrectly change the original correct result. We test with two input qubits and the random generated string is 01. Both simulations with/without error correction could find the correct answer.

On IBM's computer, we find that the real quantum computers are still error-prone. Without error correction, the results from the quantum computer do not produce a valid answer (the linear solver fails with the returned parameters); with erro correction, the result we found is 10, which is wrong.

Google's computer does not support CCNOT so we cannot run error correction.

## QAOA
Joseph
## Shor

We did not perform error correction in the case of Shor's algorithm due to the fact that the resulting circuit of Shor's algorithm with error correction could be as large as 36 qubits, which could not be handled by the current machines in both IBM or Google.

# Discuss your effort to test the programs and present results from the testing. Run each program multiple times and present statistics of the results.
## BV

We test the case when the black-box function f has its coefficients being '1000'. Below is the result from the IBM machine, where only two out of 50 trials obtain the wrong answer.

![simon_ibm](figures/bv_ibm_4.png)

## DJ

We test both the case when the black-box function f is a constant function and the case when function f is a balanced function. Here we first present the result on the IBM machine when f has its input with bit string length four and f is a balanced function. Only one out of 50 trials has the wrong output '0000'.

![simon_ibm](figures/dj_ibm.png)

Below is another test result on the IBM machine when f has its input with bit string length four and f is a constant function. In this case, all the 50 trials obtain the correct answers.

![simon_ibm](figures/dj_ibm_constant.png)

## Grover
## Simon
We are able to run on IBM and below is the result we got from IBM. It seems the computer makes a lot of errors. ![simon_ibm](figures/simon_ibm.png)

On Google, the computer still makes errors. Out of 10000 runs, only 2053 of them gave the correct answer. The most frequent results produce the wrong answer 10 (the correct string is 01).
## QAOA
Joseph
## Shor

Unfortunately we are not able to test the Shor's algorithm on the IBM machine due to the runtime constraint, where we got the exception message as follows. 

```shell
qiskit.providers.ibmq.job.exceptions.IBMQJobFailureError: 'Unable to retrieve result for job 605706d8dffd076f047a9cac. Job has failed: Circuit runtime is greater than the device repetition rate. Error code: 8020.'
```

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

To test on the IBM machine, run the following command with `-n` specifying the input bit string length.

```shell
python qiskit/bernstein.py -n $N
```

To further run on local simulation, you can add `-s` to the command.

To test on the Google machine, run the following command with `-n` specifying the input bit string length.

```shell
python cirq/bernstein.py -n $N
```

To further run on local simulation, you can add `-s` to the command.

## DJ

To test on the IBM machine, run the following command with `-n` specifying the input bit string length.

```shell
python qiskit/deutsch.py -n $N
```

To further run on local simulation, you can add `-s` to the command.

To test on the Google machine, run the following command with `-n` specifying the input bit string length.

```shell
python cirq/deutsch.py -n $N
```

To further run on local simulation, you can add `-s` to the command.

## Grover
## Simon
To test on IBM, simply run "python qiskit/simon.py" and the program will ran an example. It will first run a simulation and then connect IBM to run the quantmn computer. It will run two versions: simulated and IBM.

To test on Google, simply run "python qiskit/QAOA.py" and the program will ran an example. It will first run a simulation and then connect Google to run the quantmn computer. After we get the job id and the run finishes on Goolge, we could call check_result_with_ids() to solve the retured results.

## QAOA
To test on IBM, simply run "python qiskit/QAOA.py" and the program will ran an example. It will first run a simulation and then connect IBM to run the quantmn computer. It will run four versions: 1) simulated, 2) simulated-error-correction, 3) IBM, and 4) IBM-error-correction.

## Shor

To run the Shor's algorithm in Cirq, you can use the command "python cirq/shor.py 15"; to run the Shor's algorithm in Qiskit, you can simple use the command "python cirq/shor.py".

# Which modifications of the programs did you have to make when moving from the simulator to the quantum computer?
Work on this together. Make some bullet points first
1) Remove custom gates
2) 