datatype natural = Succ of natural | One;
exception NotNaturalNumber;

datatype 'a bstree = br of 'a bstree * 'a * 'a bstree | lf;
datatype direction = L | R;

(* 
 * Vrne seznam, ki ima na ii-tem mestu par (xi,yi)(xi​,yi​), v kateri je xi​ i-ti element seznama x, yi​ pa 
 * i-ti element seznama y. Če sta dolžini seznamov različni, vrnite pare do dolžine krajšega. 
 *)
fun zip (x: 'a list, y: 'b list) : ('a * 'b) list = 
    case (x, y) of
        ([], _) => []
      | (_, []) => []
      | (a::ax, b::bx) => (a, b) :: zip (ax, bx);

(* "Pseudoinverz" funkcije zip. *)
fun unzip (z: ('a * 'b) list) : 'a list * 'b list = 
    case z of
        [] => ([], [])
      | (a, b)::zx => let val (ax, bx) = unzip zx in (a::ax, b::bx) end;

(* Vrne naravno število, ki ustreza razliki števil a in b (a−b). Če rezultat ni naravno število, proži izjemo NotNaturalNumber. *)
fun subtract (a: natural, b: natural) : natural =
    case (a, b) of
        (One, One) => One
      | (Succ x, One) => x
      | (Succ x, Succ y) => subtract (x, y)
      | _ => raise NotNaturalNumber

(* Vrne true, če funkcija f vrne true za kateri koli element seznama s. Za prazen seznam naj vrne false. *)
fun any (f: 'a -> bool, s: 'a list) : bool = 
    case s of
        [] => false
      | x::xs => if f x then true else any (f, xs);

(* Vrne seznam elementov, preslikanih s funkcijo f vhodnega seznama s. *)
fun map (f: 'a -> 'b, s: 'a list) : 'b list = 
    case s of
        [] => []
      | x::xs => f x :: map (f, xs);

(* Vrne seznam elementov, za katere funkcija f vrne true. *)
fun filter (f: 'a -> bool, s: 'a list) : 'a list = 
    case s of
        [] => []
      | x::xs => if f x then x :: filter (f, xs) else filter (f, xs);

(* Izračuna in vrne f (... f(fz, s1), s2), ... sn). *)
fun fold (f: 'a * 'b -> 'a, z: 'a, s: 'b list) : 'a = 
    case s of
        [] => z
      | x::xs => fold (f, f (z, x), xs);

(* Vrne rotirano drevo v levo oz. desno glede na smer (L je levo, R desno), če se to da. *)
fun rotate (drevo: 'a bstree, smer: direction) : 'a bstree =
    case (drevo, smer) of
        (br (left, x, br (rightLeft, y, rightRight)), L) =>
            br (br (left, x, rightLeft), y, rightRight)
      | (br (br (leftLeft, y, leftRight), x, right), R) =>
            br (leftLeft, y, br (leftRight, x, right))
      | (_, _) => drevo

(*
fun rotate (br (left, x, br (rightLeft, y, rightRight)), L) =
      br (br (left, x, rightLeft), y, rightRight)
  | rotate (br (br (leftLeft, y, leftRight), x, right), R) =
      br (leftLeft, y, br (leftRight, x, right))
  | rotate (tree, _) = tree
*)

(* 
 * Z uporabo rotacij popravi drevo v AVL drevo z uporabo največ dveh rotacij. Poddrevesa so že AVL drevesa. 
 * V korenu se višini levega in desnega poddrevesa razlikujeta za največ dva. 
 *)
fun rebalance (drevo: 'a bstree) : 'a bstree = 
    let
        fun height lf = 0
          | height (br (l, _, r)) = 1 + Int.max (height l, height r);

        fun balanceFactor lf = 0
          | balanceFactor (br (left, _, right)) = height left - height right
    in
        case drevo of
          br (left, x, right) =>
            let
                val balance = balanceFactor drevo
            in
                if balance > 1 then
                    if balanceFactor left >= 0 then
                        rotate (drevo, R) (* Left-Left case *)
                    else
                        rotate (br (rotate (left, L), x, right), R) (* Left-Right case *)
                else if balance < ~1 then
                    if balanceFactor right <= 0 then
                        rotate (drevo, L) (* Right-Right case *)
                    else
                        rotate (br (left, x, rotate (right, R)), L) (* Right-Left case *)
                else
                    drevo (* Already balanced *)
            end
        | lf => lf
    end


(* V AVL drevo doda element e, če ga še ni. Pri tem za primerjanje elementov uporabi funkcijo c. *)
fun avl (c : 'a * 'a -> order, drevo : 'a bstree, e : 'a) : 'a bstree = 
    let
        fun insert (drevo: 'a bstree, e: 'a) : 'a bstree = 
            case drevo of
                lf => br (lf, e, lf)
              | br (l, x, r) => 
                    if c (e, x) = EQUAL then drevo
                    else if c (e, x) = LESS then rebalance (br (insert (l, e), x, r))
                    else rebalance (br (l, x, insert (r, e)))
    in
        insert (drevo, e)
    end;
