import pprint
from itertools import combinations
from typing import Dict, List, Tuple

def get_var(triplet_index: int, sequence_index: int, k: int) -> int:
    """
    Maps a triplet (x, y, z) to a unique integer variable.

    Arguments:
        triplet_index (int): Index of the triplet in the list.
        sequence_index (int): Index of the sequence.
        k (int): Length of the sequence.

    Returns:
        int: Unique variable number for the triplet (x, y, z).
    """
    return triplet_index * k + sequence_index + 1


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


def check_condition(triplet1: Tuple, triplet2: Tuple) -> bool:
    """
    Checks if the condition for the <2 relation is satisfied.

    Arguments:
        triplet1 (tuple): First triplet (x, y, z).
        triplet2 (tuple): Second triplet (u, v, w).

    Returns:
        bool: True if the condition is satisfied, False otherwise.
    """
    x, y, z = triplet1
    u, v, w = triplet2
    return (x < u and y < v) or (x < u and z < w) or (y < v and z < w)


def generate_triplets(n: int) -> List:
    """
    Generates all triplets (x, y, z) for x, y, z in {1, 2, ..., n}.

    Arguments:
        n (int): Size of the set.

    Returns:
        list: List of triplets.
    """
    return [(i, j, l) for i in range(1, n + 1) for j in range(1, n + 1) for l in range(1, n + 1)]


def generate_graph(triplets: List) -> Dict:
    """
    Generates a directed graph representing the <2 relation between triplets.
    Each triplet is a node, and an edge exists from triplet1 to triplet2
    if triplet1 <2 triplet2.
    
    Arguments:
        triplets (list): List of triplets.

    Returns:
        dict: Directed graph as an adjacency list.
    """
    graph = {}
    for i in range(len(triplets)):
        graph[triplets[i]] = []
        for j in range(len(triplets)):
            if i != j and check_condition(triplets[i], triplets[j]):
                graph[triplets[i]].append(triplets[j])
    return graph


def process_rows(triplets: List, triplet_vars: dict, k: int) -> List:
    """
    Processes the rows of the triplet matrix to generate clauses.
    Each row in the matrix must have at most one triplet (AMO).
    
    Arguments:
        triplets (list): List of triplets.
        triplet_vars (dict): Mapping of triplet to its variables.
        k (int): Length of the sequence.

    Returns:
        list: List of clauses for the rows.
    """
    clauses = []
    for triplet_index in range(len(triplets)):
        for sequence_index in range(k):
            triplet_vars[triplets[triplet_index]].append(get_var(triplet_index, sequence_index, k))
        row_vars = [get_var(triplet_index, sequence_index, k) for sequence_index in range(k)]
        clauses.extend(generate_amo(row_vars))
    return clauses


def process_columns(triplets: List, triplet_vars: Dict, k: int) -> List:
    """
    Processes the columns of the triplet matrix to generate clauses.
    Each column in the matrix must can have at least one triplet but at most one triplet (AMO + ALO).

    Arguments:
        triplets (list): List of triplets.
        triplet_vars (dict): Mapping of triplet to its variables.
        k (int): Length of the sequence.

    Returns:
        list: List of clauses for the columns.
    """
    clauses = []
    for sequence_index in range(k):
        column_vars = [get_var(triplet_index, sequence_index, k) for triplet_index in range(len(triplets))]
        clauses.append(generate_alo(column_vars))
        clauses.extend(generate_amo(column_vars))
    return clauses


def process_consecutive(triplets: List, graph: Dict,  k: int) -> List:
    """
    Processes the consecutive triplet pairs to generate clauses.
    Enforce the 2-less relation for consecutive positions in the sequence.

    Arguments:
        triplets (list): List of triplets.
        graph (dict): Directed graph representing the <2 relation.
        k (int): Length of the sequence.

    Returns:
        list: List of clauses for the consecutive triplet pairs.
    """
    clauses = []
    position_pairs = combinations(range(k), 2)
    for p, q in position_pairs:
        for i, triplet in enumerate(triplets):
            non_connected_triplets = [j for j, triplet_j in enumerate(triplets) if i != j and triplet_j not in graph[triplet]]
            for j in non_connected_triplets:
                clauses.append(f"-{get_var(i, p, k)} -{get_var(j, q, k)} 0")
    return clauses


def reduce_2less_sat(n: int, k: int) -> str:
    """
    We are given a number n, that defines a set of triplets from 1 to n
    so that  A_n = {(x, y, z) | x, y, z ∈ {1, 2, ..., n}}.

    This set is constrained by a relation that states that at least 2
    components in a triplet must be smaller than the components in 
    another triplet.

    The relation is defined as follows:
        (x, y, z) <2 (u, v, w) <=> (x < u and y < v) or (x < u and z < w) or (y < v and z < w)

    The question is, which is the longest sequence a1, a2, ..., ak (ai ∈ A_n), so that
    the rule "for each i,j, i < j : ai <2 aj" holds.
    
    The problem is transformed into a graph problem, similarly to the transformation we 
    looked at during the lab sessions. 

    The function generates a SAT problem in the DIMACS format that
    represents this problem.

    Arguments:
        n (int): Size of the set A_n.
        k (int): Length of the sequence.

    Returns:
        str: SAT problem in the DIMACS format.
    """
    triplets = generate_triplets(n)
    graph = generate_graph(triplets)
    clauses = []
    triplet_vars = {}

    triplet_vars = {triplet: [] for triplet in triplets}

    clauses.extend(process_rows(triplets, triplet_vars, k))
    clauses.extend(process_columns(triplets, triplet_vars, k))
    clauses.extend(process_consecutive(triplets, graph, k))

    num_variables = len(triplet_vars) * k
    num_clauses = len(clauses)
    dimacs_output = f"p cnf {num_variables} {num_clauses}\n" + '\n'.join(clauses)

    print("Variable map:")
    pprint.pp(triplet_vars)

    return dimacs_output
