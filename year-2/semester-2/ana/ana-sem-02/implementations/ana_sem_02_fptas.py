def run_fptas(n: int, k: int, array: list, epsilon: float = 0.4) -> int:
    """
    FPTAS (approximational) programming solution for the subset-sum problem using the Trim(L, delta) function.
    This function computes the maximum sum of a subset of the array that is equal to or smaller k.

    Args:
        n (int): The number of elements in the array.
        k (int): The sum goal.
        array (list of int): The array of integers.
        epsilon (float): The approximation factor. Default is 0.4.

    Returns:
        int: The maximum sum of a subset of the array that is less than or equal to k.
    """
    delta = epsilon / (2 * n)

    lists = [[] for _ in range(n + 1)]
    lists[0] = [0]
    for i in range(1, n + 1):
        lists[i] = merge_and_sort(lists[i - 1], [el + array[i - 1] for el in lists[i - 1]])
        lists[i] = [el for el in lists[i] if el <= k]
        lists[i] = trim(lists[i], delta)
    
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


def trim(L: list, delta: float) -> list:
    """
    Trims the list L to remove elements that are too large based on the delta value.

    Args:
        L (list): The list to be trimmed.
        delta (float): The threshold for trimming.

    Returns:
        list: The trimmed list.
    """
    Lx = [L[0]]
    last = L[0]
    for i in range(1, len(L)):
        if L[i] > last * (1 + delta):
            Lx += [L[i]]
            last = L[i]
    return Lx