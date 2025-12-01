(* settings for long expressions *)
val _ = Control.Print.printDepth := 100;
val _ = Control.Print.printLength := 1000;
val _ = Control.Print.stringDepth := 1000;
(* disable polyEq warnings *)
val _ = Control.polyEqWarn := false;


(* datatype for logical formulas *)
datatype 'a expression = 
    Not of 'a expression
|   Or of 'a expression list
|   And of 'a expression list
|   Eq of 'a expression list
|   Imp of 'a expression * 'a expression
|   Var of 'a
|   True | False;


(* linear congurence random number generator for function `prTestEq` *)
datatype 'a stream = Next of 'a * (unit -> 'a stream);

fun lcg seed =
    let fun lcg seed =
        Next (seed, fn () =>
            lcg (LargeInt.mod (1103515245 * seed + 12345, 0x7FFFFFFF)))
    in lcg (LargeInt.fromInt seed)
    end;

fun int2bool i = LargeInt.mod (i, 2) = 1;


(* conjutive normal form tester for function `satSolver` *)
fun isCNF (And es) =
    List.all
        (fn Or es => List.all (fn (Var _ | Not (Var _)) => true | _ => false) es
        |   (Var _ | Not (Var _)) => true
        |   _ => false) es
|   isCNF (Or es) = List.all (fn (Var _ | Not (Var _)) => true | _ => false) es
|   isCNF (True | False | Var _ | Not (Var _)) = true
|   isCNF _ = false;
(* exception for function `satSolver` *)
exception InvalidCNF;


(* ==================== SOME HELPER FUN. ==================== *)

(* operator for low priority right associative applications *)
infixr 1 $;
fun f $ x = f x;

(* curried equlity test *)
fun eq a b = a = b;

(* curried inequlity test *)
fun neq a b = a <> b;

(* removes all occurrences of `x` from a list *)
fun remove x = List.filter (neq x);

(* exception for nonimplemented functions *)
exception NotImplemented;


(* ==================== HELPER FUN. ==================== *)

infixr 1 $;
fun f $ x = f x;

fun eq a b = a = b;

fun neq a b = a <> b;

fun remove x = List.filter (neq x);


(* ==================== WARMUP ==================== *)

fun isolate l = 
    let
        fun aux [] _ = []
        |   aux (x::xs) seen = 
            if List.exists (eq x) seen
            then aux xs seen
            else x :: aux xs (x::seen)
    in
        aux l []
    end;

(* ==================== PART 1 ==================== *)

fun getVars expr = 
    let
        fun aux (Var v) = [v]
        |   aux (Not e) = aux e
        |   aux (Or es) = List.concat (List.map aux es)
        |   aux (And es) = List.concat (List.map aux es)
        |   aux (Eq es) = List.concat (List.map aux es)
        |   aux (Imp (e1, e2)) = aux e1 @ aux e2
        |   aux _ = []
    in
        isolate (aux expr)
    end;

fun eval es expr = 
    let
        fun aux (Var v) = List.exists (fn x => x = v) es
        |   aux (Not e) = not (aux e)
        |   aux (Or es) = List.exists aux es
        |   aux (And es) = List.all aux es
        |   aux (Eq es) = 
                (case es of
                    [] => true
                | x::xs => List.all (fn e => aux e = aux x) xs)
        |   aux (Imp (e1, e2)) = not (aux e1) orelse aux e2
        |   aux True = true
        |   aux False = false
    in
        aux expr
    end;

fun rmEmpty expr = 
    let
        fun aux (Or []) = False
        |   aux (Or [x]) = aux x
        |   aux (Or es) = Or (List.map aux es)
        |   aux (And []) = True
        |   aux (And [x]) = aux x
        |   aux (And es) = And (List.map aux es)
        |   aux (Eq []) = True
        |   aux (Eq [x]) = True
        |   aux (Eq es) = Eq (List.map aux es)
        |   aux (Not e) = Not (aux e)
        |   aux (Imp (e1, e2)) = Imp (aux e1, aux e2)
        |   aux e = e
    in
        aux expr
    end;

fun pushNegations expr = 
    let
        fun aux (Not (Not e)) = aux e
        |   aux (Not (Or es)) = And (List.map (fn e => aux (Not e)) es)
        |   aux (Not (And es)) = Or (List.map (fn e => aux (Not e)) es)
        |   aux (Not (Eq es)) = And [Or (List.map (fn e => aux (Not e)) es), Or (List.map aux es)]
        |   aux (Not (Imp (e1, e2))) = And [aux e1, aux (Not e2)]
        |   aux (Not e) = Not (aux e)
        |   aux (Or es) = Or (List.map aux es)
        |   aux (And es) = And (List.map aux es)
        |   aux (Eq es) = Eq (List.map aux es)
        |   aux (Imp (e1, e2)) = Imp (aux e1, aux e2)
        |   aux e = e
    in
       aux (rmEmpty expr)
    end;

fun rmConstants expr = 
    let
        fun neg True = False
          | neg False = True
          | neg e = Not e

        fun aux (Not e) = neg (aux e)
          | aux (Or es) = 
                let
                    val es' = List.map aux es
                in 
                    case List.exists (eq True) es' of
                        true => True
                      | false => rmEmpty (Or (remove False (remove True es')))
                end
          | aux (And es) =
                let
                    val es' = List.map aux es
                in
                    case List.exists (eq False) es' of
                        true => False
                      | false => rmEmpty (And (remove True (remove False es')))
                end
          | aux (Eq es) =
                let
                    val es' = List.map aux es
                    val vars = List.exists (fn e => not (e = True orelse e = False)) es'
                    val trues = List.exists (eq True) es'
                    val falses = List.exists (eq False) es'
                in
                    case (trues, falses, vars) of
                        (true, true, _) => False
                      | (true, false, true) => rmEmpty (And (remove False (remove True es')))
                      | (false, true, true) => rmEmpty (And (List.map neg (remove False (remove True es'))))
                      | (_, _, true) => rmEmpty (Eq es')
                      | _ => True
                end
          | aux (Imp (e1, e2)) = 
                let
                    val left = aux e1
                    val right = aux e2 
                in
                    case (left, right) of
                        (False, _) => True
                      | (_, True) => True
                      | (_, False) => neg left
                      | (True, _) => right
                      | _ => Imp (left, right)
                end
          | aux e = e
    in
        aux (rmEmpty expr)
    end;

fun rmVars expr = 
    let
        fun aux (Not e) = Not (aux e)
        |   aux (Or es) = rmEmpty (Or (isolate (List.map aux es)))
        |   aux (And es) = rmEmpty (And (isolate (List.map aux es)))
        |   aux (Eq es) = rmEmpty (Eq (isolate (List.map aux es)))
        |   aux (Imp (e1, e2)) = 
                let
                    val left = aux e1
                    val right = aux e2
                in
                    if left = right
                    then True
                    else Imp (left, right)
                end
        |   aux e = e
    in
        aux (rmEmpty expr)
    end;

 fun simplify expr = 
    let 
        fun aux e = 
            let
                val e' = rmConstants e
                val e'' = pushNegations e'
                val e''' = rmVars e''
            in
                if e = e'''
                then e
                else aux e'''
            end
    in
        aux expr
    end;

fun prTestEq seed expr1 expr2 = 
    let
        val expr1Vars = getVars expr1
        val expr2Vars = getVars expr2
        val vars = isolate (expr1Vars @ expr2Vars)

        fun nextVar (Next (x, nextStream)) [] = []
            | nextVar (Next (x, nextStream)) (v::vs) =
                let
                    val boolVal = int2bool x
                in
                    if boolVal
                    then v :: (nextVar (nextStream ()) vs)
                    else nextVar (nextStream ()) vs
                end
        
        val vars' = nextVar (lcg seed) vars
    in
        (eval vars' expr1) = (eval vars' expr2)
    end;

fun satSolver expr = 
    let
        fun neg True = False
          | neg False = True
          | neg (Not e) = e
          | neg e = Not e

        fun setVar expr var =
            let
                fun aux (Var v) = if v = var then True else Var v
                  | aux (Not (Var v)) = if v = var then False else Not (Var v)
                  | aux (Or es) = Or (List.map aux es)
                  | aux (And es) = And (List.map aux es)
                  | aux (Eq es) = Eq (List.map aux es)
                  | aux (Imp (e1, e2)) = Imp (aux e1, aux e2)
                  | aux e = e
            in
                aux expr
            end

        fun setVarNot expr var =
            let
                fun aux (Var v) = if v = var then False else Var v
                  | aux (Not (Var v)) = if v = var then True else Not (Var v)
                  | aux (Or es) = Or (List.map aux es)
                  | aux (And es) = And (List.map aux es)
                  | aux (Eq es) = Eq (List.map aux es)
                  | aux (Imp (e1, e2)) = Imp (aux e1, aux e2)
                  | aux e = e
            in
                aux expr
            end

        fun singleton (Var x) = SOME (Var x)
          | singleton (Not (Var x)) = SOME (Not (Var x))
          | singleton (Or [Var x]) = SOME (Var x)
          | singleton (Or [Not (Var x)]) = SOME (Not (Var x))
          | singleton _ = NONE

        fun rmSingletons expr assigned = 
            let
                fun aux (Var x) = if List.exists (eq x) assigned then True else Var x
                  | aux (Not (Var x)) = if List.exists (eq x) assigned then False else Not (Var x)
                  | aux (Or [Var x]) = if List.exists (eq x) assigned then True else Var x
                  | aux (Or [Not (Var x)]) = if List.exists (eq x) assigned then True else Not (Var x)
                  | aux (Or es) = Or (List.map aux es)
                  | aux (And es) = And (List.map aux es)
                  | aux (Eq es) = Eq (List.map aux es)
                  | aux (Imp (e1, e2)) = Imp (aux e1, aux e2)
                  | aux e = e
            in
                aux expr
            end

        fun dpll expr assigned = 
            let
                val expr' = rmConstants expr
                val expr'' = rmSingletons expr' assigned
            in
                case expr'' of
                    And [] => SOME assigned
                  | True => SOME assigned
                  | Or [] => NONE
                  | False => NONE
                  | _ => 
                    let
                        val vars = getVars expr''
                    in
                        case vars of
                            [] => NONE
                          | var::_ =>
                            (*let
                                val expr1 = dpll (setVar expr'' var) (var::assigned)
                                val expr2 = dpll (setVarNot expr'' var) assigned
                            in
                                case (expr1, expr2) of
                                    (SOME res1, SOME res2) => 
                                        if List.length res1 < List.length res2 then SOME res1 else SOME res2
                                  | (SOME res, _) => SOME res
                                  | (_, SOME res) => SOME res
                                  | _ => NONE
                            end*)
                            let
                                val expr1 = dpll (setVar expr'' var) (var::assigned)
                            in
                                case expr1 of
                                    SOME res => SOME res
                                  | NONE => let
                                        val expr2 = dpll (setVarNot expr'' var) assigned
                                    in
                                        case expr2 of
                                            SOME res => SOME res
                                          | NONE => NONE
                                    end
                            end
                    end
            end       
    in
        if not (isCNF expr)
        then raise InvalidCNF
        else dpll (rmEmpty expr) []
    end;

(*  Za namene terstiranja drugega dela seminarske naloge odkomentiraj
    spodnjo kodo v primeru, da funkcije satSolver nisi implementiral.
    Pred oddajo odstrani nasledji dve vrstici kode!
    Deluje samo za izraze oblike `And [Or [...], Or [...], ....]`*)

(*use "external_sat_solver.sml";
val satSolver = external_sat_solver;

*)

(* ==================== PART 2 ==================== *)

type timetable = {day : string, time: int, course: string} list;
type student = {studentID : int, curriculum : string list};

fun problemReduction _ _ _ = raise NotImplemented;

fun solutionRepresentation _ = raise NotImplemented;
