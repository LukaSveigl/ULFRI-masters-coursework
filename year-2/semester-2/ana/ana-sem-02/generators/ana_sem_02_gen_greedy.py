import os
import random


def gen_greedy(number_of_tests: int) -> None:
    """
    Generates a problem instance for the greedy solution of the subset-sum problem.

    This function heuristically creates an array of integers based on the provided parameters
    and stores it in a file. The file contains the n value in the first line, the k value in the second line,
    and the array of elements as a list of integers separated by new lines.

    Args:
        test_number (int): The test number for the problem instance.
        n (int): The number of elements in the array.
        k (int): The sum goal.
    """
    if not os.path.exists("tests/generated/greedy"):
        os.makedirs("tests/generated/greedy")

    for test_file_name in os.listdir("tests/generated/greedy"):
        os.remove(os.path.join("tests/generated/greedy", test_file_name))

    # First generate the classic greedy instances
    for test_number in range(1, number_of_tests + 1):
        n = random.randint(1000, 5000)
        array, k = gen_greedy_classic(n)
        file_name = f"tests/generated/greedy/ss_greedy_classic_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")

    # Then generate increasingly large n values
    for test_number in range(1, number_of_tests + 1):
        n = test_number * 200
        array, k = gen_greedy_classic(n)
        file_name = f"tests/generated/greedy/ss_greedy_increasing_n_{test_number}.txt"
        with open(file_name, "w+") as file:
            file.write(f"{n}\n{k}\n")
            file.write("\n".join(map(str, array)) + "\n")


def gen_greedy_classic(n: int) -> tuple:
    """
    Generates a problem instance for the greedy solution of the subset-sum problem
    where the array contains some small elements, some medium elements and some decoy large elements
    that are intended to trip up the greedy algorithm. The numbers should not be evenly distributed.
    There are more small elements than medium elements and only a few large elements.

    Args:
        n (int): The number of elements in the array.

    Returns:
        tuple: A tuple containing the generated array and the sum goal k.
    """
    k = random.randint(5000, 10000)

    ratio_large = 0.1
    ratio_small = 0.9

    large_elements = [k + 1 for _ in range(int(n * ratio_large))]
    small_elements = [k for _ in range(int(n * ratio_small))]

    k *= 2

    array = large_elements + small_elements

    random.shuffle(array)

    return array, k

    
