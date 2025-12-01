import math
import os
import random


def gen_fptas(number_of_tests: int) -> None:
    """
    Generates a problem instance for the FPTAS solution of the subset-sum problem.

    This function heuristically creates an array of integers based on the provided parameters
    and stores it in a file. The file contains the n value in the first line, the k value in the second line,
    and the array of elements as a list of integers separated by new lines.

    Args:
        test_number (int): The test number for the problem instance.
        n (int): The number of elements in the array.
        k (int): The sum goal.
    """
    if not os.path.exists("tests/generated/fptas"):
        os.makedirs("tests/generated/fptas")

    for test_file_name in os.listdir("tests/generated/fptas"):
        os.remove(os.path.join("tests/generated/fptas", test_file_name))

    # First generate the classic fptas instances
    for test_number in range(1, number_of_tests + 1):
        n = random.randint(100, 500)
        # Make sure n is even
        if n % 2 != 0:
            n += 1
        array, k = gen_fptas_classic(n)
        file_name = f"tests/generated/fptas/ss_fptas_classic_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate increasingly large n values
    for test_number in range(1, number_of_tests + 1):
        n = test_number * 200
        array, k = gen_fptas_classic(n)
        file_name = f"tests/generated/fptas/ss_fptas_increasing_n_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")


def gen_fptas_classic(n: int) -> tuple:
    """
    Generates a problem instance for the FPTAS solution of the subset-sum problem.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum goal k.
    """
    # Generate elements close to each other
    base_value = random.randint(10, 100)
    array = [base_value + random.randint(-10, 10) for _ in range(n)]

    # Set k to be slightly less than the sum of the array
    while True:
        k = sum(array) - random.randint(1, 50)
        if k > 0:
            break

    return array, k