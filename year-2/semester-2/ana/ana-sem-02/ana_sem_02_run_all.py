import os
import sys
import time

import pandas as pd

from generators.ana_sem_02_gen_dyn import gen_dyn
from generators.ana_sem_02_gen_exh import gen_exh
from generators.ana_sem_02_gen_greedy import gen_greedy
from generators.ana_sem_02_gen_fptas import gen_fptas

from implementations.ana_sem_02_dyn import run_dyn
from implementations.ana_sem_02_exh import run_exh
from implementations.ana_sem_02_greedy import run_greedy
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
    algorithms = {
        'dyn': (run_dyn, gen_dyn), 
        'exh': (run_exh, gen_exh),
        'greedy': (run_greedy, gen_greedy),
        'fptas': (run_fptas, gen_fptas)
    }
    algorithm = 'dyn'
    number_of_tests = 10

    if len(sys.argv) == 3:
        algorithm = sys.argv[1]
        number_of_tests = int(sys.argv[2])
        
        if algorithm not in algorithms:
            print(f"Invalid algorithm '{algorithm}'. Choose from {list(algorithms.keys())}.")
            sys.exit(1)
    else:
        print("Invalid number of arguments. Usage: python ana_sem_02_run_all.py <algorithm> <number_of_tests>")
        sys.exit(1)

    run, gen = algorithms[algorithm]
    gen(number_of_tests)

    print(f"Generated {number_of_tests} tests for algorithm '{algorithm}'")
    print(f"Running with algorithm '{algorithm}'")

    # For every file in the generated tests, run the algorithm and store the results in a CSV file.
    # The file is stored in the results directory with the name <algorithm>_results.csv.
    # The file contains the following columns: test_type, n, k, result, time_taken, memory_used.
    results = []
    for file_name in os.listdir(f"tests/generated/{algorithm}"):
        if file_name.endswith(".txt"):
            file_path = os.path.join(f"tests/generated/{algorithm}", file_name)
            print(f"Running test {file_name}")

            n, k, values = load_data(file_path)

            start = time.time()
            result, aux = run(n, k, values)
            end = time.time()

            test_type = '_'.join(file_name.removesuffix('.txt').split('_')[1:-1])

            results.append({
                'test_type': test_type,
                'n': n,
                'k': k,
                'result': result,
                'time_taken': end - start,
                'memory_used': sys.getsizeof(aux if aux is not None else 0)
            })

    df = pd.DataFrame(results)
    df.to_csv(f"results/{algorithm}_results.csv", index=False)
    print(f"Results saved to results/{algorithm}_results.csv")
    