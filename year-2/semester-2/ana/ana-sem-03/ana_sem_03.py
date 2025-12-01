import os
import random
import sys

import networkx as nx


def exact_cut(edges: list) -> int:
    """
    Computes the exact minimum cut size for the given edges and nodes.
    
    Args:
        edges (list): List of edges in the graph.
        nodes (dict): Dictionary of nodes in the graph.

    Returns:
        int: The size of the cut.
    """
    G = nx.Graph()
    G.add_edges_from(edges)
    return len(nx.minimum_edge_cut(G))


def find(node: str, nodes: dict[str, str]) -> str:
    """
    Finds the representative of the set containing the given node using path compression.
    
    Args:
        node (str): The node to find.
        nodes (dict[str, str]): Dictionary of nodes in the graph.

    Returns:
        str: The representative of the set containing the node.
    """
    root = nodes[node]

    if nodes[root] != root:
        nodes[node] = find(root, nodes)
        return nodes[node]
    
    return root


def union(node1: str, node2: str, nodes: dict[str, str], group_sizes: list[int]) -> bool:
    """
    Unites the sets containing the two nodes if they are not already in the same set.

    Args:
        node1 (str): The first node.
        node2 (str): The second node.
        nodes (dict[str, str]): Dictionary of nodes in the graph.
        group_sizes (list[int]): List of sizes of each group.

    Returns:
        bool: True if the sets were united, False if they were already in the same set.
    """
    root1 = find(node1, nodes)
    root2 = find(node2, nodes)

    if root1 == root2:
        return False
    
    size1 = group_sizes[int(root1)]
    size2 = group_sizes[int(root2)]

    if size1 < size2:
        nodes[root1] = root2
        group_sizes[int(root2)] += size1
    else:
        nodes[root2] = root1
        group_sizes[int(root1)] += size2

    return True


def karger(edges: list, nodes: dict[str, str]) -> tuple[int, int]:
    """
    Implements Karger's algorithm to find a minimum cut in the given graph using the union-find method.
    
    Args:
        edges (list): List of edges in the graph.
        nodes (dict[str, str]): Dictionary of nodes in the graph.

    Returns:
        tuple: A tuple containing the size of the minimum cut and the number of repetitions performed.
    """
    group_sizes = [1] * (max([int(node) for node in nodes.keys()]) + 1)
    shuffle_edges = edges[:]
    random.shuffle(shuffle_edges)
    
    repetitions = 0
    for u, v in shuffle_edges:
        if union(u, v, nodes, group_sizes):
            repetitions += 1
            if repetitions == len(nodes.keys()) - 2:
                break
    
    cut_size = sum(1 for edge in edges if find(edge[0], nodes) != find(edge[1], nodes))
    return cut_size
    

def run_algorithm(edges: list, nodes: dict[str, str], max_reps: int, exact_optimum: int) -> tuple[list, int, float]:
    """
    Runs the Karger's algorithm on the given graph and returns the results. The results include the minimum cut size and the 
    average number of repetitions performed until the minimum cut was found.

    Args:
        edges (list): List of edges in the graph.
        nodes (dict[str, str]): Dictionary of nodes in the graph.
        max_reps (int): Maximum number of repetitions to run the algorithm.
        exact_optimum (int): The exact minimum cut size for comparison.

    Returns:
        tuple: A tuple containing the minimum cut size and the average number of repetitions.
    """
    results = []
    repetitions = []
    repetitions_count = 0
    optimum = 0

    for rep in range(max_reps):
        if rep % 100 == 0:
            print(f"Running repetition {rep + 1} of {max_reps}...")
            
        for i in range(1, 1001):
            optimum = karger(edges[:], nodes.copy())
            repetitions_count = i
            if optimum == exact_optimum:
                break
        results.append((optimum, repetitions_count))
        repetitions.append(repetitions_count)

    best_optimum = min(result[0] for result in results)
    average_reps = sum(repetitions) / len(repetitions) if repetitions else 0

    return results, best_optimum, average_reps


def load_data(file_path: str) -> tuple[list, dict[str, str]]:
    """
    Loads the data (first node, second node) from a file and returns the edges 
    and the nodes dict used for the union-find.

    Args:
        file_path (str): Path to the input file.

    Returns:
        tuple: A tuple containing a list of edges and a dict of nodes.
    """
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    edges = []
    nodes = set()

    for line in lines:
        if line.strip():
            u, v = line.strip().split(' ')
            edges.append((u, v))
            nodes.update([u, v])

    nodes = {node: node for node in nodes}
    return edges, nodes


def store_data(file_path: str, dimensions: tuple, results: list, best_optimum: int, average_reps: float):
    """
    Stores the results of the algorithm in a file. First, it stores all the results (optimum and repetitions) in the file, 
    then it outputs a divider line, and finally it stores graph name, graph dimensions, best optimum, and average repetitions.

    Args:
        file_path (str): Path to the output file.
        dimensions (tuple): Dimensions of the graph (number of nodes, number of edges).
        results (list): List of tuples containing the optimum and repetitions for each run.
        best_optimum (int): The best optimum found.
        average_reps (float): The average number of repetitions performed.
    """
    with open(file_path, 'w') as file:
        for result in results:
            file.write(f"{result[0]} {result[1]}\n")
        
        file.write("-" * 50 + "\n")
        file.write(f"Graph: {os.path.basename(file_path)}\n")
        file.write(f"Dimensions (n, m): {dimensions[0]}, {dimensions[1]}\n")
        file.write(f"Best Optimum: {best_optimum}\n")
        file.write(f"Average Repetitions: {average_reps:.2f}\n")


if __name__ == '__main__':
    file_path = 'tests/g01.graph'
    output_path = 'outputs/g01.out'
    
    if len(sys.argv) == 2:
        file_path = sys.argv[1]

        if not os.path.exists(file_path):
            print(f"File '{file_path}' does not exist.")
            sys.exit(1)    

        output_path = 'outputs/' + file_path.split('/')[-1].replace('.graph', '.out')
    else:
        print("Invalid number of arguments. Usage: python ana_sem_03.py <file_path>")
        sys.exit(1)

    print(f"Running the algorithm on file '{file_path}', storing results in '{output_path}'")
    edges, nodes = load_data(file_path)

    dimensions = (len(nodes.keys()), len(edges))

    print(f"Graph dimensions: {dimensions[0]} nodes, {dimensions[1]} edges")

    exact = exact_cut(edges)
    print(f"Exact cut size: {exact}")

    max_reps = 1000  
    results, best_optimum, average_reps = run_algorithm(edges, nodes, max_reps, exact)

    print(f"Best optimum found: {best_optimum}")
    print(f"Average repetitions: {average_reps:.2f}")

    store_data(output_path, dimensions, results, best_optimum, average_reps)