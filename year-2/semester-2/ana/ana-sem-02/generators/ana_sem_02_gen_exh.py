import os
import random

import sympy as sp


def gen_exh(number_of_tests: int) -> None:
    """
    Generates a problem instance for the exhaustive solution of the subset-sum problem.

    This function heuristically creates an array of integers based on the provided parameters
    and stores it in a file. The file contains the n value in the first line, the k value in the second line,
    and the array of elements as a list of integers separated by new lines.

    Args:
        test_number (int): The test number for the problem instance.
        n (int): The number of elements in the array.
        k (int): The sum goal.
    """
    if not os.path.exists("tests/generated/exh"):
        os.makedirs("tests/generated/exh")

    for test_file_name in os.listdir("tests/generated/exh"):
        os.remove(os.path.join("tests/generated/exh", test_file_name))

    # First generate close gap instances
    for test_number in range(1, number_of_tests + 1):
        n = random.randint(1000, 5000)
        array, k = gen_exh_close_gaps(n)
        file_name = f"tests/generated/exh/ss_close_gap_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate prime instances
    # for test_number in range(1, number_of_tests + 1):
    #     n = random.randint(1000, 5000)
    #     array, k = gen_exh_primes(n)
    #     file_name = f"tests/generated/exh/ss_primes_{test_number}.txt"
    #     with open(file_name, "w+") as file:
    #         file.write(f"{n}\n{k}\n")
    #         file.write("\n".join(map(str, array)) + "\n")

    # Then generate files with increasingly large n values
    for test_number in range(1, number_of_tests + 1):
        n = test_number * 200
        array, k = gen_exh_close_gaps(n)
        file_name = f"tests/generated/exh/ss_increasing_n_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")


def gen_exh_close_gaps(n):
    """
    Generates a problem instance for the exhaustive solution of the subset-sum problem
    where the numbers are small and close to each other.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    array = [random.randint(1, 10) for _ in range(n)]
    k = random.randint(1, 10 * n)
    return array, k


def gen_exh_primes(n):
    """
    Generates a problem instance for the exhaustive solution of the subset-sum problem
    where the numbers are prime numbers to avoid repeated sums from number patterns.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    # Randomly sample n primes from the first n * 100 primes
    primes = list(sp.primerange(1, n * 10))
    array = random.sample(primes, n)  
    random.shuffle(array)
    k = sum(array) // 4
    return array, k
