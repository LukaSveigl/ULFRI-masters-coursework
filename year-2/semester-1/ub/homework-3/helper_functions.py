"""Helper functions for HW3"""
import numpy as np
from copy import deepcopy
from matplotlib.axes import Axes


class Node:
    def __init__(
        self,
        name: str,
        left: "Node",
        left_distance: float,
        right: "Node",
        right_distance: float,
        confidence: float = None,
    ):
        """A node in a binary tree produced by neighbor joining algorithm.

        Parameters
        ----------
        name: str
            Name of the node.
        left: Node
            Left child.
        left_distance: float
            The distance to the left child.
        right: Node
            Right child.
        right_distance: float
            The distance to the right child.
        confidence: float
            The confidence level of the split determined by the bootstrap method.
            Only used if you implement Bonus Problem 1.

        Notes
        -----
        The current public API needs to remain as it is, i.e., don't change the
        names of the properties in the template, as the tests expect this kind
        of structure. However, feel free to add any methods/properties/attributes
        that you might need in your tree construction.

        """
        self.name = name
        self.left = left
        self.left_distance = left_distance
        self.right = right
        self.right_distance = right_distance
        self.confidence = confidence


def neighbor_joining(distances: np.ndarray, labels: list) -> Node:
    """The Neighbor-Joining algorithm.

    For the same results as in the later test dendrograms;
    add new nodes to the end of the list/matrix and
    in case of ties, use np.argmin to choose the joining pair.

    Parameters
    ----------
    distances: np.ndarray
        A 2d square, symmetric distance matrix containing distances between
        data points. The diagonal entries should always be zero; d(x, x) = 0.
    labels: list
        A list of labels corresponding to entries in the distances matrix.
        Use them to set names of nodes.

    Returns
    -------
    Node
        A root node of the neighbor joining tree.

    """

    def compute_q_matrix(distances_matrix: np.ndarray) -> tuple:
        """
        Computes the r, Q matrix and the indices of the nodes to merge.

        Parameters
        ----------
        distances_matrix: np.ndarray
            The distances matrix.
        
        Returns
        -------
        tuple
            A tuple containing the r values, the Q matrix and the indices of the nodes to merge.
        """
        r = np.sum(distances_matrix, axis=1)
        q_matrix = np.zeros((len(distances_matrix), len(distances_matrix[0])))

        min_q = ''
        min_i = 0
        min_j = 0

        for i in range(len(distances_matrix)):
            for j in range(i + 1, len(distances_matrix[0])):
                q_matrix[i, j] = (len(distances_matrix) - 2) * distances_matrix[i, j] - r[i] - r[j]

                if min_q == '' or q_matrix[i, j] < min_q:
                    min_q = q_matrix[i, j]
                    min_i = i
                    min_j = j

        return r, q_matrix, min_i, min_j

    def compute_distances_matrix(distances_matrix: np.ndarray, min_i: int, min_j: int) -> np.ndarray:
        """
        Computes the new distances matrix.

        Parameters
        ----------
        distances_matrix: np.ndarray
            The distances matrix.
        min_i: int
            The index of the first node to merge.
        min_j: int
            The index of the second node to merge.

        Returns
        -------
        np.ndarray
            The new distances matrix.
        """
        new_distances_matrix = np.zeros((len(distances_matrix) - 1, len(distances_matrix[0]) - 1))
        new_i = -1
        for i in range(len(distances_matrix) + 1):
            if i == min_i or i == min_j:
                continue

            new_i += 1
            new_j = -1

            for j in range(len(distances_matrix[0]) + 1):
                if j == min_i or j == min_j:
                    continue

                new_j += 1

                if j == i:
                    continue
                
                if j == len(distances_matrix[0]):
                    new_distances_matrix[new_i, new_j] = (distances_matrix[i, min_i] + distances_matrix[i, min_j] - distances_matrix[min_i, min_j]) / 2
                elif i == len(distances_matrix):
                    new_distances_matrix[new_i, new_j] = (distances_matrix[min_i, j] + distances_matrix[min_j, j] - distances_matrix[min_i, min_j]) / 2
                else:
                    new_distances_matrix[new_i, new_j] = distances_matrix[i, j]

        return new_distances_matrix

    def update_labels(labels: list, min_i: int, min_j: int) -> list:
        """
        Generate new labels for the nodes after merging two nodes.

        Parameters
        ----------
        labels: list
            The list of labels of the nodes.
        min_i: int
            The index of the first node to merge.
        min_j: int
            The index of the second node to merge.

        Returns
        -------
        list
            The new list of labels.
        """
        new_labels = [label for idx, label in enumerate(labels) if idx != min_i and idx != min_j]
        new_labels.append(f'Node {node_counter}')
        return new_labels

    distances_matrix = np.copy(distances)

    nodes = {label: Node(label, None, 0, None, 0) for label in labels}
    node_counter = 0

    # Iterate until there are only two nodes left - after that, the root node is created.
    while len(distances_matrix) > 2:

        # 1. Compute the Q matrix.
        r, q_matrix, min_i, min_j = compute_q_matrix(distances_matrix)
                
        # 2. Compute the L(Ti, v) and L(Tj, v) values and create a new node.
        L_i = (distances_matrix[min_i, min_j] / 2) + ((r[min_i] - r[min_j]) / (2 * (len(distances_matrix) - 2)))
        L_j = (distances_matrix[min_i, min_j] / 2) + ((r[min_j] - r[min_i]) / (2 * (len(distances_matrix) - 2)))

        nodes[f'Node {node_counter}'] = Node(
            f'Node {node_counter}', 
            nodes[labels[min_i]], 
            L_i, 
            nodes[labels[min_j]], 
            L_j
        )

        # 3. Update the distances matrix.
        distances_matrix = compute_distances_matrix(distances_matrix, min_i, min_j)

        # 4. Update the labels.
        labels = update_labels(labels, min_i, min_j)

        node_counter += 1

    # Create the root node and connect the first two nodes.
    root = Node(
        'Root',
        nodes[labels[0]],
        distances_matrix[0, 1] / 2,
        nodes[labels[1]],
        distances_matrix[0, 1] / 2
    )

    return root


def plot_nj_tree(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.
        But make sure your function can work without them, for testing purposes.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree(tree=tree, ax=ax)
    >>> fig.savefig("example.png")
    """

    def compute_layout(node: Node, distance: int = 0) -> tuple:
        """
        Computes the layout of the tree, defined as number of leaves and max distance.

        Parameters
        ----------
        node: Node
            The current node.
        distance: int
            The current distance from the root node.

        Returns
        -------
        tuple
            A tuple containing the number of leaves and the max distance.
        """
        if node.left is None and node.right is None:
            return 1, distance

        left_leaves, left_distance = compute_layout(node.left, distance + node.left_distance)
        right_leaves, right_distance = compute_layout(node.right, distance + node.right_distance)

        return left_leaves + right_leaves, max(left_distance, right_distance)

    def plot_tree(
            ax: Axes, 
            node: Node, 
            root: str, 
            max_distance: int, 
            leaf_count: int, 
            current_distance: int, 
            current_leaf: int
        ) -> tuple:
        """
        Recursively plots the dendrogram of the tree.

        Parameters
        ----------
        ax: Axes
            The matplotlib Axes object.
        node: Node
            The current node.
        root: str
            The name of the root node.
        max_distance: int
            The maximum distance of the tree.
        leaf_count: int
            The number of leaves in the tree.
        current_distance: int
            The current distance from the root node.
        current_leaf: int
            The current leaf number.

        Returns
        -------
        tuple
            A tuple containing the current leaf and the current y position.
        """
        if node is None:
            return current_leaf, None

        current_leaf, left_y = plot_tree(
            ax, node.left, root, max_distance, leaf_count, current_distance + node.left_distance, current_leaf
        )
        current_leaf, right_y = plot_tree(
            ax, node.right, root, max_distance, leaf_count, current_distance + node.right_distance, current_leaf
        )

        # If the node is a leaf node, draw only the name.
        if node.left is None and node.right is None:
            if 'taxonomy_map' in kwargs and 'taxonomy_color_map' in kwargs and 'name_to_accession_map' in kwargs:
                #color = kwargs['taxonomy_color_map'][kwargs['taxonomy_map'][node.name]]
                color = kwargs['taxonomy_color_map'][kwargs['taxonomy_map'][kwargs['name_to_accession_map'][node.name]]]
            else:
                color = kwargs.get('color', 'black')
            ax.text(
                current_distance + 0.2, current_leaf, node.name, verticalalignment='center', fontsize=kwargs.get('node_fontsize', 12), color=color
            )
            current_y = current_leaf
            current_leaf += 1
        # If the node is not a leaf, draw all the connecting lines and distances.
        else:
            current_y = (left_y + right_y) / 2

            # Draw the beginning line of the graph - as the first split to the nodes occurs at 0, this line
            # is drawn from -0.2 to 0.
            if node.name == root:
                ax.plot([-0.2, 0], [current_y, current_y], color=kwargs.get('color', 'black'))#color='black')

            if node.confidence is not None:
                ax.text(
                    current_distance, current_y, f"{node.confidence}%", verticalalignment='center', fontsize=kwargs.get('dist_fontsize', 8)
                )

            # Draw the vertical line from the current node to the children.
            ax.plot([current_distance, current_distance], [left_y, right_y], color=kwargs.get('color', 'black'))

            # Draw the lines connecting the nodes.
            ax.plot(
                [current_distance, current_distance + node.left_distance], [left_y, left_y], color=kwargs.get('color', 'black'), lw=kwargs.get('lw', 1.5)
            )
            ax.plot(
                [current_distance, current_distance + node.right_distance], [right_y, right_y], color=kwargs.get('color', 'black'), lw=kwargs.get('lw', 1.5)
            )

            # Draw the distances right next to the lines - at the beginning of the line.
            ax.text(current_distance + 0.1, left_y, f"{round(node.left_distance, 2)}", verticalalignment='bottom', fontsize=kwargs.get('dist_fontsize', 8))
            ax.text(current_distance + 0.1, right_y, f"{round(node.right_distance, 2)}", verticalalignment='bottom', fontsize=kwargs.get('dist_fontsize', 8))

        return current_leaf, current_y

    leaf_count, max_distance = compute_layout(tree)

    # Set the axis parameters.
    padding = 0.2
    ax.set_ylim(-padding, leaf_count - 1 + padding)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plot_tree(ax, tree, tree.name, max_distance, leaf_count, 0, 0)

    return ax


def _find_a_parent_to_node(tree: Node, node: Node) -> tuple:
    """Utility function for reroot_tree"""
    stack = [tree]

    while len(stack) > 0:

        current_node = stack.pop()
        if node.name == current_node.left.name:
            return current_node, "left"
        elif node.name == current_node.right.name:
            return current_node, "right"

        stack += [
            n for n in [current_node.left, current_node.right] if n.left is not None
        ]

    return None


def _remove_child_from_parent(parent_node: Node, child_location: str) -> None:
    """Utility function for reroot_tree"""
    setattr(parent_node, child_location, None)
    setattr(parent_node, f"{child_location}_distance", 0.0)


def reroot_tree(original_tree: Node, outgroup_node: Node) -> Node:
    """A function to create a new root and invert a tree accordingly.

    This function reroots tree with nodes in original format. If you
    added any other relational parameters to your nodes, these parameters
    will not be inverted! You can modify this implementation or create
    additional functions to fix them.

    Parameters
    ----------
    original_tree: Node
        A root node of the original tree.
    outgroup_node: Node
        A Node to set as an outgroup (already included in a tree).
        Find it by it's name and then use it as parameter.

    Returns
    -------
    Node
        Inverted tree with a new root node.
    """
    tree = deepcopy(original_tree)

    parent, child_loc = _find_a_parent_to_node(tree, outgroup_node)
    distance = getattr(parent, f"{child_loc}_distance")
    _remove_child_from_parent(parent, child_loc)

    new_root = Node("new_root", parent, distance / 2, outgroup_node, distance / 2)
    child = parent

    while tree != child:
        parent, child_loc = _find_a_parent_to_node(tree, child)

        distance = getattr(parent, f"{child_loc}_distance")
        _remove_child_from_parent(parent, child_loc)

        empty_side = "left" if child.left is None else "right"
        setattr(child, f"{empty_side}_distance", distance)
        setattr(child, empty_side, parent)

        if tree.name == parent.name:
            break
        child = parent

    other_child_loc = "right" if child_loc == "left" else "left"
    other_child_distance = getattr(parent, f"{other_child_loc}_distance")

    setattr(child, f"{empty_side}_distance", other_child_distance + distance)
    setattr(child, empty_side, getattr(parent, other_child_loc))

    return new_root


def sort_children_by_leaves(tree: Node) -> None:
    """Sort the children of a tree by their corresponding number of leaves.

    The tree can be changed inplace.

    Paramteres
    ----------
    tree: Node
        The root node of the tree.

    """
    def count_leaves(node: Node) -> int:
        """
        Counts the number of leaves in the tree.

        Parameters
        ----------
        node: Node
            The current node.

        Returns
        -------
        int
            The number of leaves in the tree.
        """
        if node is None:
            return 0
        if node.left is None and node.right is None:
            return 1
        return count_leaves(node.left) + count_leaves(node.right)
    
    if tree is None:
        return
    
    leaves_left = count_leaves(tree.left)
    leaves_right = count_leaves(tree.right)

    if leaves_left > leaves_right:
        tree.left, tree.right = tree.right, tree.left
        tree.left_distance, tree.right_distance = tree.right_distance, tree.left_distance

    sort_children_by_leaves(tree.left)
    sort_children_by_leaves(tree.right)


def plot_nj_tree_radial(tree: Node, ax: Axes = None, **kwargs) -> None:
    """A function for plotting neighbor joining phylogeny dendrogram
    with a radial layout.

    Parameters
    ----------
    tree: Node
        The root of the phylogenetic tree produced by `neighbor_joining(...)`.
    ax: Axes
        A matplotlib Axes object which should be used for plotting.
    kwargs
        Feel free to replace/use these with any additional arguments you need.

    Example
    -------
    >>> import matplotlib.pyplot as plt
    >>>
    >>> tree = neighbor_joining(distances)
    >>> fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 8))
    >>> plot_nj_tree_radial(tree=tree, ax=ax)
    >>> fig.savefig("example_radial.png")

    """
    raise NotImplementedError()


def global_alignment(seq1, seq2, scoring_function):
    """Global sequence alignment using the Needleman–Wunsch algorithm.

    Indels should be denoted with the "-" character.

    Parameters
    ----------
    seq1: str
        First sequence to be aligned.
    seq2: str
        Second sequence to be aligned.
    scoring_function: Callable

    Returns
    -------
    str
        First aligned sequence.
    str
        Second aligned sequence.
    float
        Final score of the alignment.

    Examples
    --------
    >>> global_alignment("abracadabra", "dabarakadara", lambda x, y: [-1, 1][x == y])
    ('-ab-racadabra', 'dabarakada-ra', 5.0)

    Other alignments are not possible.

    """

    # Initialize the scoring matrix.
    scoring_matrix = np.zeros((len(seq1) + 1, len(seq2) + 1))

    # Fill the scoring matrix.
    for i in range(1, len(seq1) + 1):
        scoring_matrix[i, 0] = scoring_matrix[i - 1, 0] + scoring_function(seq1[i - 1], "-")
    for j in range(1, len(seq2) + 1):
        scoring_matrix[0, j] = scoring_matrix[0, j - 1] + scoring_function("-", seq2[j - 1])
    for i in range(1, len(seq1) + 1):
        for j in range(1, len(seq2) + 1):
            scoring_matrix[i, j] = max(
                scoring_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1]),
                scoring_matrix[i - 1, j] + scoring_function(seq1[i - 1], "-"),
                scoring_matrix[i, j - 1] + scoring_function("-", seq2[j - 1])
            )

    # Traceback.
    i, j = len(seq1), len(seq2)
    aligned_seq1, aligned_seq2 = "", ""
    score = scoring_matrix[i, j]
    while i > 0 or j > 0:
        if i > 0 and j > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j - 1] + scoring_function(seq1[i - 1], seq2[j - 1]):
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            i -= 1
            j -= 1
        elif i > 0 and scoring_matrix[i, j] == scoring_matrix[i - 1, j] + scoring_function(seq1[i - 1], "-"):
            aligned_seq1 = seq1[i - 1] + aligned_seq1
            aligned_seq2 = "-" + aligned_seq2
            i -= 1
        else:
            aligned_seq1 = "-" + aligned_seq1
            aligned_seq2 = seq2[j - 1] + aligned_seq2
            j -= 1

    return aligned_seq1, aligned_seq2, score #scoring_matrix[-1, -1]