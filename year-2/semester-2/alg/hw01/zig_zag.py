class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None


class ZigZag:
    def __init__(self, root):
        self.tree = root
    
    def show(self):
        """
        Prints the tree in zig-zag order.
        """
        if self.tree is None:
            return
        
        # Create two stacks to store the nodes.
        current_level = []
        next_level = []

        # Add the root node to the current level.
        current_level.append(self.tree)

        # Set the direction to left to right.
        left_to_right = True

        # Loop until the current level is empty.
        while len(current_level) > 0:
            node = current_level.pop(-1)
            print(node.value, end=' ')

            # If the direction is left to right, add the left child first.
            if left_to_right:
                if node.left is not None:
                    next_level.append(node.left)
                if node.right is not None:
                    next_level.append(node.right)
            else:
                # If the direction is right to left, add the right child first.
                if node.right is not None:
                    next_level.append(node.right)
                if node.left is not None:
                    next_level.append(node.left)

            # If the current level is empty, swap the current level with the next level.
            if len(current_level) == 0:
                current_level, next_level = next_level, current_level
                left_to_right = not left_to_right


if __name__ == '__main__':
    root = BinaryTree(1)
    root.left = BinaryTree(2)
    root.right = BinaryTree(3)
    root.left.left = BinaryTree(4)
    root.left.right = BinaryTree(5)
    root.right.left = BinaryTree(6)
    root.right.right = BinaryTree(7)

    zigzag = ZigZag(root)
    zigzag.show()
        

        