import sys
import time

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
        'dyn': run_dyn, 
        'exh': run_exh, 
        'greedy': run_greedy, 
        'fptas': run_fptas
    }
    algorithm = 'dyn'

    if len(sys.argv) > 2:
        algorithm = sys.argv[1]
        
        if algorithm not in algorithms:
            print(f"Invalid algorithm '{algorithm}'. Choose from {list(algorithms.keys())}.")
            sys.exit(1)
    else:
        print("Invalid number of arguments. Usage: python ana_sem_02.py <algorithm> <file_path>")
        sys.exit(1)

    print(f"Running with algorithm '{algorithm}' on file '{sys.argv[2]}'")

    n, k, values = load_data(sys.argv[2])
    run = algorithms[algorithm]
    
    start = time.time()
    result, aux = run(n, k, values)
    end = time.time()

    print(f"Result: {result}")
    print(f"Time taken: {end - start:.6f} seconds")
    print(f"Memory used: {sys.getsizeof(aux if aux is not None else 0)} bytes")
    print("-" * 40)
    # Flush the output buffer to ensure all output is printed
    sys.stdout.flush()
