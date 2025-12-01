import os
import random


def gen_dyn(number_of_tests: int) -> None:
    """
    Generates a problem instance for the dynamic programming solution of the subset-sum problem.

    This function heuristically creates an array of integers based on the provided parameters
    and stores it in a file. The file contains the n value in the first line, the k value in the second line,
    and the array of elements as a list of integers separated by new lines.

    Args:
        test_number (int): The test number for the problem instance.
        n (int): The number of elements in the array.
        k (int): The sum goal.
    """
    if not os.path.exists("tests/generated/dyn"):
        os.makedirs("tests/generated/dyn")

    for test_file_name in os.listdir("tests/generated/dyn"):
        os.remove(os.path.join("tests/generated/dyn", test_file_name))

    # First generate no match instances
    for test_number in range(1, number_of_tests + 1):
        n = random.randint(1000, 5000)
        array, k = gen_dyn_no_match(n)
        file_name = f"tests/generated/dyn/ss_no_match_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate small elements instances
    for test_number in range(1, number_of_tests + 1):
        n = random.randint(1000, 5000)
        array, k = gen_dyn_small_elements(n)
        file_name = f"tests/generated/dyn/ss_small_elem_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate files with increasingly large n values
    for test_number in range(1, number_of_tests + 1):
        n = test_number * 200
        array, k = gen_dyn_small_elements(n)
        file_name = f"tests/generated/dyn/ss_increasing_n_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate density one instances
    # for test_number in range(1, test_ratio + 1):
    #     n = random.randint(1000, 5000)
    #     array, k = gen_dyn_density_one(n)
    #     file_name = f"tests/generated/dyn/ss_density_one_{test_number}.txt"
    #     with open(file_name, "w+") as file:
    #         file.write(f"{n}\n{k}\n")
    #         file.write("\n".join(map(str, array)) + "\n")

    # Then generate negative instances
    # for test_number in range(1, test_ratio + 1):
    #     n = random.randint(1000, 5000)
    #     array, k = gen_dyn_negative(n)
    #     file_name = f"tests/generated/dyn/ss_negative_{test_number}.txt"
    #     with open(file_name, "w+") as file:
    #         file.write(f"{n}\n{k}\n")
    #         file.write("\n".join(map(str, array)) + "\n")


def gen_dyn_no_match(n):
    """
    Generates a problem instance for the dynamic programming solution of the subset-sum problem
    where no subset matches the sum k. This is intended to stress the algorithm by maxing out the
    DP table.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    array = [random.randint(1, 20) for _ in range(n)]
    k = sum(array) + random.randint(1, 20)
    return array, k


def gen_dyn_density_one(n):
    """
    Generates a problem instance for the dynamic programming solution of the subset-sum problem
    where the density of the array is 1. The goal here is to generate many small, overlapping elements
    that can be combined to reach the sum k.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    max_a = int(2 ** n)
    array = [random.randint(1, max_a) for _ in range(n)]
    k = sum(array) // 2
    return array, k


def gen_dyn_small_elements(n):
    """
    Generates a problem instance for the dynamic programming solution of the subset-sum problem
    where the elements are small integers. This is intended to create a scenario where the algorithm
    has to process many elements to find the optimal solution.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    array = [random.randint(1, 20) for _ in range(n)]
    k = sum(array) // 2
    return array, k


def gen_dyn_negative(n):
    """
    Generates a problem instance for the dynamic programming solution of the subset-sum problem
    where the elements can be negative integers. This is intended to create a scenario where the algorithm
    has to process both positive and negative elements to find the optimal solution.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum k.
    """
    array = [random.randint(-20, 20) for _ in range(n)]
    k = sum(array) // 2
    return array, k