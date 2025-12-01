def run_exh(n: int, k: int, array: list) -> int:
    """
    Exhaustive programming solution for the subset-sum problem.
    This function computes the maximum sum of a subset of the array that is equal to k.

    Args:
        n (int): The number of elements in the array.
        k (int): The sum goal.
        array (list of int): The array of integers.

    Returns:
        int: The maximum sum of a subset of the array that is less than or equal to k.
    """
    lists = [[] for _ in range(n + 1)]
    lists[0] = [0]
    for i in range(1, n + 1):
        lists[i] = merge_and_sort(lists[i - 1], [el + array[i - 1] for el in lists[i - 1]])
        lists[i] = [el for el in lists[i] if el <= k]
    return lists[-1][-1] if lists[-1] else 0, lists


def merge_and_sort(list1: list, list2: list) -> list:
    """
    Merges two lists and sorts the result.

    Args:
        list1 (list): The first list.
        list2 (list): The second list.

    Returns:
        list: The merged and sorted list.
    """
    merged = list1 + list2
    return sorted(set(merged))