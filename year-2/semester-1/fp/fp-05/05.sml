structure Rational =
struct
    datatype rational = Frac of int * int | Whole of int

    exception BadRational

    fun simplify (a, b) = 
        let
            fun gcd(a, 0) = abs a
              | gcd(a, b) = gcd(b, a mod b)

            val d = gcd(a, b)
            val (a', b') = (a div d, b div d)
        in
            if b' = 1 then Whole a'
            else if b' = ~1 then Whole (~a')
            else Frac (a', b')
        end

    (*fun makeRational (a, b) =
        if b = 0 then raise BadRational
        else if b = 1 then Whole a
        else if b = ~1 then Whole (~a)
        else if a = 0 then Whole 0
        else if b < 0 then Frac (~a, ~b)
        else Frac (a, b)*)
    fun makeRational (a, b) =
        if b = 0 then raise BadRational
        else if a = 0 then Whole 0
        else if b < 0 then simplify (~a, ~b)
        else simplify (a, b)

    fun neg (Whole a) = Whole (~a)
      | neg (Frac (a, b)) = Frac (~a, b)

    fun inv (Whole a) = makeRational (1, a)
      | inv (Frac (a, b)) = makeRational (b, a)

    fun add (Whole a, Whole b) = Whole (a+b)
      | add (Whole a, Frac (c, d)) = makeRational (a*d + c, d)
      | add (Frac (a, b), Whole c) = makeRational (a + c*b, b)
      | add (Frac (a, b), Frac (c, d)) = if b = d then makeRational (a+c, b) else makeRational (a*d + c*b, b*d)

    fun mul (Whole a, Whole b) = Whole (a*b)
      | mul (Whole a, Frac (c, d)) = makeRational (a*c, d)
      | mul (Frac (a, b), Whole c) = makeRational (a*c, b)
      | mul (Frac (a, b), Frac (c, d)) = makeRational (a*c, b*d)

    fun toString (Whole a) = Int.toString a
      | toString (Frac (a, b)) = Int.toString a ^ "/" ^ Int.toString b
end

signature EQ =
sig
    type t
    val eq : t -> t -> bool
end

signature SET =
sig
    (* podatkovni tip za elemente množice *)
    type item

    (* podatkovni tip množico *)
    type set

    (* prazna množica *)
    val empty : set

    (* vrne množico s samo podanim elementom *)
    val singleton : item -> set

    (* unija množic *)
    val union : set -> set -> set

    (* razlika množic (prva - druga) *)
    val difference : set -> set -> set

    (* a je prva množica podmnožica druge *)
    val subset : set -> set -> bool
end

funsig SETFN (Eq : EQ) = SET

functor SetFn (Eq : EQ) : SET =
struct
    type item = Eq.t

    type set = item list

    val empty = []

    fun singleton x = [x]

    fun union xs ys = xs @ ys

    fun difference xs ys = List.filter (fn x => not (List.exists (fn y => Eq.eq x y) ys)) xs

    fun subset xs ys = List.all (fn x => List.exists (fn y => Eq.eq x y) ys) xs
end