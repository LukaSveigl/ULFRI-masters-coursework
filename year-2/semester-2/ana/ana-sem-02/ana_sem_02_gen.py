import sys

from generators.ana_sem_02_gen_dyn import gen_dyn
from generators.ana_sem_02_gen_exh import gen_exh
from generators.ana_sem_02_gen_greedy import gen_greedy
from generators.ana_sem_02_gen_fptas import gen_fptas


if __name__ == '__main__':
    algorithm = 'dyn'
    algorithms = {
        'dyn': gen_dyn, 
        'exh': gen_exh, 
        'greedy': gen_greedy, 
        'fptas': gen_fptas
    }
    number_of_tests = 10

    if len(sys.argv) == 3:
        algorithm = sys.argv[1]
        number_of_tests = int(sys.argv[2])
        
        if algorithm not in algorithms:
            print(f"Invalid algorithm '{algorithm}'. Choose from {list(algorithms.keys())}.")
            sys.exit(1)

    else:
        print("Invalid number of arguments. Usage: python ana_sem_02.py <algorithm> <number_of_tests>")
        sys.exit(1)

    gen = algorithms[algorithm]
    print(f"Generating with algorithm '{algorithm}' for number_of_tests = {number_of_tests}")

    gen(number_of_tests)
    print(f"Generated {number_of_tests} elements for algorithm '{algorithm}'")