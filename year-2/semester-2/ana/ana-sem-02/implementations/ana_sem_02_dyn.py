def run_dyn(n: int, k: int, array: list) -> int:
    """
    Dynamic programming solution for the subset-sum problem.
    This function computes the maximum sum of a subset of the array that is equal to k.

    Args:
        n (int): The number of elements in the array.
        k (int): The sum goal.
        array (list of int): The array of integers.

    Returns:
        int: The maximum sum of a subset of the array that is less than or equal to k.
    """
    # Initialize a DP table with dimensions (n+1) x (k+1)
    dp = [[0] * (k + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(k + 1):
            # When filling out the DP table, we use the following logic:
            # S(i, j) = max(S(i-1, j), S(i-1, j - a_i) + a_i)
            # S(0, j) = 0 for all j (base case)
            # S(i, 0) = 0 for all i (base case)
            if array[i - 1] <= j:
                dp[i][j] = max(dp[i - 1][j], dp[i - 1][j - array[i - 1]] + array[i - 1])
            else:
                dp[i][j] = dp[i - 1][j]

    return dp[n][k], dp