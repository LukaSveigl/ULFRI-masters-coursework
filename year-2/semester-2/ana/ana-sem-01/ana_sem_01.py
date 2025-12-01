import sys

from ana_sem_01_nqueens import reduce_nq_sat
from ana_sem_01_2less import reduce_2less_sat

if __name__ == '__main__':
    problem = 'nqueens'

    if len(sys.argv) > 1:
        problem = sys.argv[1]
        if problem not in ['nqueens', '2less']:
            print(f"Unknown problem: {problem}, must be 'nqueens' or '2less'")
            sys.exit(1)

    if problem == 'nqueens':
        print("Enter the size of the chessboard (n):")
        n = int(input())

        if n < 1:
            print("Invalid size. Must be a positive integer.")
            sys.exit(1)

        print(reduce_nq_sat(n))

    elif problem == '2less':
        print("Enter (n):")
        n = int(input())

        if n < 1:
            print("Invalid size. Must be a positive integer.")
            sys.exit(1)

        print("Enter (k):")
        k = int(input())

        if k < 1:
            print("Invalid size. Must be a positive integer.")
            sys.exit(1)

        output = reduce_2less_sat(n, k)

        # Write the output to a file.
        with open("2less_output.dimacs", "w") as file:
            file.write(output)

    else:
        print("Unknown problem. Must be 'nqueens' or '2less'")
        sys.exit(1)