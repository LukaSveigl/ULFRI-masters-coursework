import os
import sys
import time

import pandas as pd 
import matplotlib.pyplot as plt

from implementations.ana_sem_02_fptas import run_fptas


def load_data(file_path: str) -> tuple:
    """
    Loads the data (n, k, array of values) from a file.
    The first line contains n, the second line contains k, and the third line contains the array of values.

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple: A tuple containing n (int), k (int), and the array of values (list of int).
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()

    n = int(lines[0].strip())
    k = int(lines[1].strip())
    values = [int(line.strip()) for line in lines[2:]]

    return n, k, values


if __name__ == '__main__':
    # Generate epsilons with values from 0.1 to 2.0 with a step of 0.3
    #epsilons: list[float] = []
    #i = 0.1
    #while i <= 2.0:
    #    epsilons.append(i)
    #    i += 0.3
    epsilons = [0.1 * i for i in range(1, 21)]


    file_path = 'tests/generated/fptas/ss_fptas_increasing_n_20.txt'

    results = []

    for epsilon in epsilons:
        print(f"Running FPTAS with epsilon = {epsilon}")

        # Load the data from the file
        n, k, values = load_data(file_path)

        start = time.time()
        result, aux = run_fptas(n, k, values, epsilon)
        end = time.time()

        results.append({
            'epsilon': epsilon,
            'n': n,
            'k': k,
            'result': result,
            'time': end - start
        })

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    df.to_csv('outputs/fptas_results_epsilon.csv', index=False)

    # Plot the results in terms of time taken vs epsilon and store the plot
    # into a pdf file
    plt.figure(figsize=(6, 4))
    plt.plot(df['epsilon'], df['time'], marker='o', label='Time Taken')
    plt.title('Time Taken vs Epsilon for FPTAS')
    plt.xlabel('Epsilon')
    plt.ylabel('Time Taken (seconds)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/fptas_time_vs_epsilon.pdf')
    plt.close()