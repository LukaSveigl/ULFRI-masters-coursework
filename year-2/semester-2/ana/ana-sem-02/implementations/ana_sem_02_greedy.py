def run_greedy(n: int, k: int, array: list) -> int:
    """
    Greedy (approximational) programming solution for the subset-sum problem.
    This function computes the maximum sum of a subset of the array that is equal to or smaller k.

    Args:
        n (int): The number of elements in the array.
        k (int): The sum goal.
        array (list of int): The array of integers.

    Returns:
        int: The maximum sum of a subset of the array that is less than or equal to k.
    """
    # Sort the array in descending order
    array = sorted(array, reverse=True)

    # Initialize the sum and the result
    current_sum = 0

    # Iterate through the sorted array and add elements to the sum until it exceeds k
    for value in array:
        if current_sum + value <= k:
            current_sum += value

    return current_sum, None  # No auxiliary data is generated in this case