use "03.sml";

Control.printWarnings := false;

(* izpis daljših izrazov v interpreterju *)
val _ = Control.Print.printDepth := 100;
val _ = Control.Print.printLength := 1000;
val _ = Control.Print.stringDepth := 1000;

val _ = print "~~~~~~~~ zip ~~~~~~~~\n";
val zip_test1 = (zip ([1,2,3], ["a","b","c"]) = [(1,"a"), (2,"b"), (3,"c")]); (* Equal length lists *)
val zip_test2 = (zip ([1,2,3], ["a","b"]) = [(1,"a"), (2,"b")]); (* First list longer *)
val zip_test3 = (zip ([1], ["a","b","c"]) = [(1, "a")]); (* Second list longer *)
val zip_test4 = (zip ([], ["a","b","c"]) = []); (* First list empty *)
val zip_test5 = (zip ([1,2,3], []) = []); (* Second list empty *)
val zip_test6 = (zip ([], []) = []); (* Both lists empty *)

val _ = print "~~~~~~~~ unzip ~~~~~~~~\n";
val unzip_test1 = (unzip [(1,"a"), (2,"b"), (3,"c")] = ([1,2,3], ["a","b","c"]));
val unzip_test2 = (unzip [] = ([], [])); (* Empty list case *)

val _ = print "~~~~~~~~ subtract ~~~~~~~~\n";
val nat1 = Succ (Succ (Succ One)); (* Natural number representation for 3 *)
val nat2 = Succ (Succ One); (* Natural number representation for 2 *)
val nat3 = One; (* Natural number representation for 1 *)
val subtract_test1 = (subtract (nat1, nat2) = Succ One); (* 3 - 2 = 1 *)
(*val subtract_test2 = (subtract (nat2, nat1) = nat3); (* 2 - 3 raises NotNaturalNumber exception *)*)
val subtract_test3 = (subtract (nat1, nat1) = One); (* 3 - 3 = 0 *)

val _ = print "~~~~~~~~ any ~~~~~~~~\n";
val any_test1 = (any (fn x => x > 5, [1, 2, 6]) = true); (* At least one element is greater than 5 *)
val any_test2 = (any (fn x => x < 0, [1, 2, 3]) = false); (* No element is less than 0 *)
val any_test3 = (any (fn x => x mod 2 = 0, []) = false); (* Empty list should return false *)

val _ = print "~~~~~~~~ map ~~~~~~~~\n";
val map_test1 = (map (fn x => x * x, [1, 2, 3, 4]) = [1, 4, 9, 16]); (* Square each element *)
val map_test2 = (map (fn x => x + 1, []) = []); (* Empty list should return empty list *)

val _ = print "~~~~~~~~ filter ~~~~~~~~\n";
val filter_test1 = (filter (fn x => x mod 2 = 0, [1, 2, 3, 4, 5, 6]) = [2, 4, 6]); (* Filter even numbers *)
val filter_test2 = (filter (fn x => x < 0, [1, 2, 3, 4, 5, 6]) = []); (* No elements satisfy the predicate *)
val filter_test3 = (filter (fn x => x > 3, [1, 2, 3, 4, 5, 6]) = [4, 5, 6]); (* Filter numbers greater than 3 *)
val filter_test4 = (filter (fn x => x > 3, []) = []); (* Empty list should return empty list *)

val _ = print "~~~~~~~~ fold ~~~~~~~~\n";
val fold_test1 = (fold (op +, 0, [1, 2, 3, 4]) = 10); (* Sum of elements *)
val fold_test2 = (fold (op *, 1, [1, 2, 3, 4]) = 24); (* Product of elements *)
val fold_test3 = (fold (op +, 0, []) = 0); (* Empty list should return initial value *)

val _ = print "~~~~~~~~ rotate ~~~~~~~~\n";
val bst1 = br (lf, 2, br (lf, 3, lf));
val bst2 = br (br (lf, 1, lf), 2, lf);
val rotate_test1 = (rotate (bst1, L) = br (br (lf, 2, lf), 3, lf)); (* Left rotation *)
val rotate_test2 = (rotate (bst2, R) = br (lf, 1, br (lf, 2, lf))); (* Right rotation *)
val rotate_test3 = (rotate (lf, L) = lf); (* Rotate empty tree *)
val rotate_test4 = (rotate (lf, R) = lf); (* Rotate empty tree *)

val _ = print "~~~~~~~~ rebalance ~~~~~~~~\n";
val unbalanced_bst = br (br (lf, 1, lf), 2, br (lf, 3, br (lf, 4, lf)));
val balanced_bst = br (br (lf, 1, lf), 2, br (lf, 3, lf)); (* Manually balanced tree *)
val rebalance_test1 = (rebalance unbalanced_bst = balanced_bst);

fun height lf = 0
  | height (br (l, _, r)) = 1 + Int.max (height l, height r);

(* izpis drevesa po nivojih *)
fun showTree (toString : 'a -> string, t : 'a bstree) =
let fun strign_of_avltree_level (lvl, t) = case t of  
        lf => if lvl = 0 then "nil" else "   "
    |   br (l, n, r) =>
        let val make_space = String.map (fn _ => #" ")
            val sn = toString n
            val sl = strign_of_avltree_level (lvl, l)
            val sr = strign_of_avltree_level (lvl, r)
        in if height t = lvl
            then make_space sl ^ sn ^ make_space sr
            else sl ^ make_space sn ^ sr
        end
    fun print_levels lvl =
        if lvl >= 0
        then (print (Int.toString lvl ^ ": " ^ strign_of_avltree_level (lvl, t) ^ "\n");
                    print_levels (lvl - 1))
        else ()
  in  print_levels (height t)
end;

(* primeri vstavljanja elementov v AVL drevo *)
fun avlInt (t, i) = avl (Int.compare, t, i);
fun showTreeInt t = showTree(Int.toString, t);


(* Rotate trees output. *)
val _ = print "~~~~~~~~ rotate trees ~~~~~~~~\n";
val tree = br (lf, 2, br (lf, 3, lf));
val _ = showTreeInt tree;
val tree = rotate (tree, L);
val _ = showTreeInt tree;
(*A more complicated tree.*)
val tree = br (br (lf, 1, lf), 2, br (lf, 3, br (lf, 4, lf)));
val _ = showTreeInt tree;
val tree = rotate (tree, L);
val _ = showTreeInt tree;

(* Rebalance trees output. *)
val _ = print "~~~~~~~~ rebalance trees ~~~~~~~~\n";
(* Create a very big (10 elements) and unbalanced tree. *)
val unbalanced_bst = fold (fn (z, x) => avl (Int.compare, z, x), lf, [1, 2, 3, 4, 5, 6, 7, ~4, ~3, ~2, ~1, 0]);
val _ = showTreeInt unbalanced_bst;
val _ = showTreeInt (rebalance unbalanced_bst);

val tr = lf : int bstree;
val _ = showTreeInt tr;
val tr = avlInt (tr, 1);
val _ = showTreeInt tr;
val tr = avlInt (tr, 2);
val _ = showTreeInt tr;
val tr = avlInt (tr, 3);
val _ = showTreeInt tr;
val tr = avlInt (tr, 4);
val _ = showTreeInt tr;
val tr = avlInt (tr, 5);
val _ = showTreeInt tr;
val tr = avlInt (tr, 6);
val _ = showTreeInt tr;
val tr = avlInt (tr, 7);
val _ = showTreeInt tr;
val tr = avlInt (tr, ~4);
val _ = showTreeInt tr;
val tr = avlInt (tr, ~3);
val _ = showTreeInt tr;
val tr = avlInt (tr, ~2);
val _ = showTreeInt tr;
val tr = avlInt (tr, ~1);
val _ = showTreeInt tr;
val tr = avlInt (tr, 0);
val _ = showTreeInt tr;

val from0to13 = fold (fn (z, x) => avl (Int.compare, z, x), lf, List.tabulate (14, fn i => i));
