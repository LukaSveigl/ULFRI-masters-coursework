from typing import Dict, List

def get_var(i: int, j: int, n: int, variable_map: Dict) -> int:
    """
    Maps a chessboard position (i, j) to a unique integer variable.

    Arguments:
        i (int): Row index.
        j (int): Column index.

    Returns:
        int: Unique variable number for the position (i, j).
    """
    if (i, j) not in variable_map:
        variable_map[(i, j)] = i * n + j + 1
    return variable_map[(i, j)]


def generate_alo(vars: List) -> str:
    """
    Generates a clause that ensures at least one variable is true (ALO).

    Arguments:
        vars (list): List of variable names.

    Returns:
        str: A string representing the clause in DIMACS format.
    """
    return ' '.join(map(str, vars)) + ' 0'


def generate_amo(vars: List) -> str:
    """
    Generates clauses that ensure at most one variable is true (AMO).

    Arguments:
        vars (list): List of variable names.

    Returns:
        list: A list of strings representing the clauses in DIMACS format.
    """
    return [f'-{vars[i]} -{vars[j]} 0' for i in range(len(vars)) for j in range(i + 1, len(vars))]


def process_rows(n: int, variable_map: Dict) -> List:
    """
    Processes the rows of the chessboard to ensure that each row has exactly one queen.

    Arguments:
        n (int): Size of the chessboard.
        variable_map (dict): A mapping of chessboard positions to variable numbers.

    Returns:
        list: A list of clauses for the rows.
    """
    clauses = []
    for i in range(n):
        row_vars = [get_var(i, j, n, variable_map) for j in range(n)]
        clauses.append(generate_alo(row_vars))
        clauses.extend(generate_amo(row_vars))
    return clauses


def process_columns(n: int, variable_map: Dict) -> List:
    """
    Processes the columns of the chessboard to ensure that each column has exactly one queen.

    Arguments:
        n (int): Size of the chessboard.
        variable_map (dict): A mapping of chessboard positions to variable numbers.

    Returns:
        list: A list of clauses for the columns.
    """
    clauses = []
    for j in range(n):
        col_vars = [get_var(i, j, n, variable_map) for i in range(n)]
        clauses.append(generate_alo(col_vars))
        clauses.extend(generate_amo(col_vars))
    return clauses


def process_diagonals(n: int, variable_map: Dict) -> List:
    """
    Processes the diagonals of the chessboard to ensure that each diagonal has at most one queen.

    Arguments:
        n (int): Size of the chessboard.
        variable_map (dict): A mapping of chessboard positions to variable numbers.

    Returns:
        list: A list of clauses for the diagonals.
    """
    clauses = []
    for d in range(-n + 1, n):
        left_diag_vars = []
        right_diag_vars = []
        for i in range(n):
            j_left = i + d
            j_right = i - d
            if 0 <= j_left < n:
                left_diag_vars.append(get_var(i, j_left, n, variable_map))
            if 0 <= j_right < n:
                right_diag_vars.append(get_var(i, j_right, n, variable_map))
        if len(left_diag_vars) > 1:
            clauses.extend(generate_amo(left_diag_vars))
        if len(right_diag_vars) > 1:
            clauses.extend(generate_amo(right_diag_vars))
    return clauses


def reduce_nq_sat(n: int) -> str:
    """
    Accepts the size of the chessboard and outputs a SAT problem in the DIMACS format.

    This function generates clauses for the N-Queens problem, ensuring that:
    1. Each row has exactly one queen (ALO + AMO).
    2. Each column has exactly one queen (ALO + AMO).
    3. Each diagonal has at most one queen (AMO).
    
    Arguments:
        n (int): Size of the chessboard.

    Returns:
        str: SAT problem in the DIMACS format.
    """
    clauses = []
    variable_map = {}

    clauses = process_rows(n, variable_map)
    clauses.extend(process_columns(n, variable_map))
    clauses.extend(process_diagonals(n, variable_map))
    
    # Generate the DIMACS format output.
    dimacs_output = f"p cnf {len(variable_map)} {len(clauses)}\n"
    dimacs_output += '\n'.join(clauses)

    return dimacs_output
