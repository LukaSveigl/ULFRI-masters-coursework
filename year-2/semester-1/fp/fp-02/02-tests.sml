use "02.sml";

(* Test for neg function *)
val _ = print "~~~~~~~~ neg ~~~~~~~~\n";
val test_type: number -> number = neg;
val test1 = (neg Zero = Zero); (* negating Zero should return Zero *)
val test2 = (neg (Succ (Succ Zero)) = Pred (Pred Zero)); (* negating two successors *)
val test3 = (neg (Pred (Pred Zero)) = Succ (Succ Zero)); (* negating two predecessors *)


(* Test for add function *)
val _ = print "~~~~~~~~ add ~~~~~~~~\n";
val test_type: number * number -> number = add;
val test1 = (add (Succ Zero, Pred Zero) = Zero); (* adding one and negative one *)
val test2 = (add (Succ Zero, Succ Zero) = Succ (Succ Zero)); (* adding two positive numbers *)
val test3 = (add (Pred Zero, Pred Zero) = Pred (Pred Zero)); (* adding two negative numbers *)


(* Test for comp function *)
val _ = print "~~~~~~~~ comp ~~~~~~~~\n";
val test_type: number * number -> order = comp;
val test1 = (comp (Succ Zero, Zero) = GREATER); (* 1 > 0 *)
val test2 = (comp (Zero, Succ Zero) = LESS); (* 0 < 1 *)
val test3 = (comp (Zero, Zero) = EQUAL); (* 0 == 0 *)


(* Test for contains function *)
val _ = print "~~~~~~~~ contains ~~~~~~~~\n";
val test_type: tree * int -> bool = contains;
val tree1 = Node (5, Leaf 3, Leaf 7);
val test1 = (contains (tree1, 5) = true); (* 5 is in the tree *)
val test2 = (contains (tree1, 6) = false); (* 6 is not in the tree *)
val test3 = (contains (tree1, 7) = true); (* 7 is in the tree *)

(* Test for countLeaves function *)
val _ = print "~~~~~~~~ countLeaves ~~~~~~~~\n";
val test_type: tree -> int = countLeaves;
val test1 = (countLeaves (Leaf 3) = 1); (* single leaf *)
val test2 = (countLeaves (Node (5, Leaf 3, Leaf 7)) = 2); (* two leaves in the tree *)
val test3 = (countLeaves (Node (5, Node (3, Leaf 1, Leaf 2), Leaf 7)) = 3); (* tree with 3 leaves *)

(* Test for countBranches function *)
val _ = print "~~~~~~~~ countBranches ~~~~~~~~\n";
val test_type: tree -> int = countBranches;
val test1 = (countBranches (Leaf 3) = 0); (* no branches in a leaf *)
val test2 = (countBranches (Node (5, Leaf 3, Leaf 7)) = 1); (* one branch in the tree *)
val test3 = (countBranches (Node (5, Node (3, Leaf 1, Leaf 2), Leaf 7)) = 2); (* two branches in the tree *)
val test4 = (countBranches (Node (5, Node (3, Leaf 1, Leaf 2), Node (7, Leaf 6, Leaf 8))) = 3); (* Explanation: Three branches at the root node and two child nodes *)
val test5 = (countBranches (Leaf 42) = 0); (* Explanation: Leaf has no branches *)
val test6 = (countBranches (Node (1, Node (2, Node (3, Leaf 4, Leaf 5), Leaf 6), Leaf 7)) = 3); (* Explanation: Three branches despite the unbalanced structure *)
val test7 = (countBranches (Node (1, Leaf 2, Node (3, Leaf 4, Node (5, Leaf 6, Leaf 7)))) = 3); (* Explanation: Another unbalanced structure with three branches *)

(* Test for height function *)
val _ = print "~~~~~~~~ height ~~~~~~~~\n";
val test_type: tree -> int = height;
val test1 = (height (Leaf 3) = 1); (* height of a single leaf is 1 *)
val test2 = (height (Node (5, Leaf 3, Leaf 7)) = 2); (* tree with two levels has height 2 *)
val test3 = (height (Node (5, Node (3, Leaf 1, Leaf 2), Leaf 7)) = 3); (* tree with three levels has height 3 *)

(* Test for toList function *)
val _ = print "~~~~~~~~ toList ~~~~~~~~\n";
val test_type: tree -> int list = toList;
val test1 = (toList (Leaf 3) = [3]); (* single leaf converts to a list with one element *)
val test2 = (toList (Node (5, Leaf 3, Leaf 7)) = [3, 5, 7]); (* in-order traversal of a simple tree *)
val test3 = (toList (Node (5, Node (3, Leaf 1, Leaf 2), Leaf 7)) = [1, 3, 2, 5, 7]); (* more complex tree traversal *)

(* Test for isBalanced function *)
val _ = print "~~~~~~~~ isBalanced ~~~~~~~~\n";
val test_type: tree -> bool = isBalanced;
val test1 = (isBalanced (Leaf 3) = true); (* a single leaf is balanced *)
val test2 = (isBalanced (Node (5, Leaf 3, Leaf 7)) = true); (* perfectly balanced tree *)
val test3 = (isBalanced (Node (5, Node (3, Leaf 1, Leaf 2), Leaf 7)) = true); (* balanced but more complex tree *)
val test4 = (isBalanced (Node (5, Node (3, Leaf 1, Leaf 2), Node (7, Leaf 6, Node (8, Leaf 9, Leaf 10)))) = false); (* unbalanced tree *)

(* Test for isBST function *)
val _ = print "~~~~~~~~ isBST ~~~~~~~~\n";
val test_type: tree -> bool = isBST;
val test1 = (isBST (Leaf 3) = true); (* a single leaf is a valid BST *)
val test2 = (isBST (Node (5, Leaf 3, Leaf 7)) = true); (* valid binary search tree *)
val test3 = (isBST (Node (5, Node (7, Leaf 6, Leaf 8), Leaf 3)) = false); (* invalid binary search tree *)