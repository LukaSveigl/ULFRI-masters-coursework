datatype number = Zero | Succ of number | Pred of number;

(* Negira število a. Pretvorba v int ni dovoljena! *)
fun neg (a : number) : number =
    case a of
        Zero => Zero
    | Succ n => Pred (neg n)
    | Pred n => Succ (neg n);

(* Vrne vsoto števil a in b. Pretvorba v int ni dovoljena! *)
fun add (a : number, b : number) : number =
    case a of
        Zero => b
    | Succ n => Succ (add (n, b))
    | Pred n => Pred (add (n, b));

(* Vrne rezultat primerjave števil a in b. Pretvorba v int ter uporaba funkcij `add` in `neg` ni dovoljena!
    namig: uporabi funkcijo simp *)
fun comp (a : number, b : number) : order = 
    let
        fun simp Zero = Zero
            | simp (Succ n) =
                (case simp n of
                    Pred m => m
                    | m => Succ m)
            | simp (Pred n) =
                (case simp n of
                    Succ m => m
                    | m => Pred m)
    in 
        case simp (add (a, neg b)) of
            Zero => EQUAL
        | Succ _ => GREATER
        | Pred _ => LESS
    end

datatype tree = Node of int * tree * tree | Leaf of int;

(* Vrne true, če drevo vsebuje element x. *)
fun contains (tree : tree, x : int) : bool = 
    case tree of
        Leaf n => n = x
    | Node (n, l, r) => n = x orelse contains (l, x) orelse contains (r, x);

(* Vrne število listov v drevesu. *)
fun countLeaves (tree : tree) : int = 
    case tree of 
        Leaf _ => 1
        | Node (_, l, r) => countLeaves (l) + countLeaves (r);

(* Vrne število število vej v drevesu. *)
fun countBranches (tree : tree) : int = 
    case tree of 
        Leaf _ => 0
        | Node (_, l, r) => 2 + countBranches (l) + countBranches (r);

(* Vrne višino drevesa. Višina lista je 1. *)
fun height (tree : tree) : int = 
    case tree of 
        Leaf _ => 1
        | Node (_, l, r) => 1 + Int.max (height (l), height (r));

(* Pretvori drevo v seznam z vmesnim prehodom (in-order traversal). *)
fun toList (tree : tree) : int list = 
    case tree of 
        Leaf n => [n]
        | Node (n, l, r) => toList (l) @ [n] @ toList (r);

(* Vrne true, če je drevo uravnoteženo:
 * - Obe poddrevesi sta uravnoteženi.
 * - Višini poddreves se razlikujeta kvečjemu za 1.
 * - Listi so uravnoteženi po definiciji.
 *)
fun isBalanced (tree : tree) : bool = 
    let
        fun isBalanced' (tree : tree) : (bool * int) = 
            case tree of 
                Leaf _ => (true, 1)
                | Node (_, l, r) => 
                    let
                        val (lBalanced, lHeight) = isBalanced' (l)
                        val (rBalanced, rHeight) = isBalanced' (r)
                    in
                        (lBalanced andalso rBalanced andalso abs (lHeight - rHeight) <= 1, 1 + Int.max (lHeight, rHeight))
                    end
    in
        #1 (isBalanced' (tree))
    end

(* Vrne true, če je drevo binarno iskalno drevo:
 * - Vrednosti levega poddrevesa so strogo manjši od vrednosti vozlišča.
 * - Vrednosti desnega poddrevesa so strogo večji od vrednosti vozlišča.
 * - Obe poddrevesi sta binarni iskalni drevesi.
 * - Listi so binarna iskalna drevesa po definiciji.
 *)
fun isBST (tree : tree) : bool = 
    let
        fun isBST' (tree : tree) : (bool * int * int) = 
            case tree of 
                Leaf n => (true, n, n)
                | Node (n, l, r) => 
                    let
                        val (lBST, lMin, lMax) = isBST' (l)
                        val (rBST, rMin, rMax) = isBST' (r)
                    in
                        (lBST andalso rBST andalso lMax < n andalso n < rMin, lMin, rMax)
                    end
    in
        #1 (isBST' (tree))
    end