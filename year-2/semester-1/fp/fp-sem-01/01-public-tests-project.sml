use "01-project.sml";
val _ = print "\n\n========== running public tests... ==========\n";

val _ = Control.Print.printDepth := 100;
val _ = Control.Print.printLength := 1000;
val _ = Control.Print.stringDepth := 1000;
val _ = Control.polyEqWarn := false;


(* WIP tests ! *)

val all_tests : bool list ref = ref [];

(* ==================== WARMUP ==================== *)

val _ = print "---------- isolate ----------\n";
val _ : ''a list -> ''a list = isolate;
val test1 = isolate [1,2,34,6,7,4,5,5,3,4,5,6,2,1,3,0] =
        [1,2,34,6,7,4,5,3,0] handle NotImplemented => false;
val _ = (all_tests := [test1]);


(* ==================== PART 1 ==================== *)


val exp1 = (Eq [True, Eq [False, Var 3, And [And [],
                Or [Var 1, Not (Eq [Var 4, False, True])], Imp (True, Var 2)]]]);
val exp2 = And [Not (Var 3), Not (Var 2)];
val exp3 = And [Not (Var 1), Not (Var 2)];
val exp4 = Eq [True, Eq [False, Var 3, And [And [Var 2], Eq [Var 0], Or [Var 6, Var 3],
            Or [Var 1, Not (Eq [Var 4, False, True])], Imp (Eq [Or []], Var 8)]]];
val exp5 = Eq [True, Eq [False, Var 2, And [And [Var 2], Eq [Var 0], Or [Var 6, Var 3],
            Or [Var 1, Not (Eq [Var 2, False, True])], Imp (Eq [Or []], Var 2)]]];

val exp6 = (And [Or [Not (Var 1), Not (Var 5), Var 7, Var 7], Or [Var 1, Var 4, Not (Var 7)], Or [Not (Var 1), Var 6, Not (Var 7), Var 7, Var 7], Or [Var 2, Var 5, Not (Var 7), Var 7], Or [Not (Var 4), Var 7, Var 7, Var 7], Or [Var 7, Var 7, Var 7, Not (Var 7)], Or [Var 2, Not (Var 7), Var 7], Or [Not (Var 3), Not (Var 1), Var 7, Var 7], Or [Var 5, Var 4, Not (Var 7)], Or [Var 3, Var 6, Not (Var 7), Var 7], Or [Var 3, Var 1, Var 5, Not (Var 7)], Or [Var 2, Var 5, Not (Var 7)], Or [Not (Var 1), Not (Var 4), Var 7], Or [Not (Var 2), Not (Var 4), Var 7, Var 7], Or [Not (Var 2), Not (Var 5), Not (Var 7), Var 7], Or [Not (Var 3), Not (Var 6), Not (Var 7), Not (Var 7)], Or [Var 3, Var 2, Not (Var 7)], Or [Var 5, Var 6], Or [Not (Var 2), Not (Var 5), Var 7, Var 7, Var 7], Or [Var 3, Var 6], Or [Var 1, Var 5, Var 4, Not (Var 7)], Or [Not (Var 3), Not (Var 5), Var 7, Not (Var 7), Not (Var 7)], Or [Not (Var 3), Var 1, Not (Var 5), Not (Var 7), Not (Var 7)], Or [Var 2, Var 6], Or [Var 1, Not (Var 7)], Or [Not (Var 6), Not (Var 5), Not (Var 7), Not (Var 7)], Or [Var 5, Not (Var 7), Var 7], Or [Not (Var 2), Var 5, Var 7, Not (Var 7)], Or [Not (Var 6), Not (Var 4), Var 7, Not (Var 7)], Or [Var 3, Var 2, Var 1, Not (Var 7)], Or [Not (Var 3), Var 1, Not (Var 6), Var 4, Not (Var 7), Not (Var 7)], Or [Var 6, Not (Var 7)], Or [Var 3, Not (Var 7)], Or [Var 2, Var 3], Or [Not (Var 2), Not (Var 1), Not (Var 6), Not (Var 5), Not (Var 7), Not (Var 7)], Or [Not (Var 2), Var 5, Var 7], Or [Var 3, Var 5], Or [Not (Var 3), Not (Var 2), Not (Var 7), Not (Var 7)], Or [Not (Var 2), Not (Var 5), Var 7, Not (Var 7), Var 7, Not (Var 7)], Or [Var 2, Not (Var 5), Var 7], Or [Var 2, Var 6, Var 4, Not (Var 7)], Or [Var 2, Not (Var 5), Var 7, Not (Var 7)], Or [Var 6, Var 5, Not (Var 7)], Or [Var 4, Not (Var 7)], Or [Var 3, Var 6, Not (Var 7), Not (Var 7)]]);


val _ = print "---------- getVars ----------\n";
val _ : ''a expression -> ''a list = getVars;
val test1 = getVars (Eq [Var "A", Var "B", Imp (Var "D", Not (Var "Q")),
                            Var "D", Var "B"]) = ["A","B","D","Q"] handle NotImplemented => false;
val test2 = getVars (Imp (Imp (True, Var 1), Imp (Var 0, Not (Var 4)))) = [1,0,4] handle NotImplemented => false;
val test3 = getVars exp1 = [3,1,4,2] handle NotImplemented => false;
val test4 = getVars exp2 = [3,2] handle NotImplemented => false;
val test5 = getVars True = [] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3, test4, test5]);

val _ = print "---------- eval ----------\n";
val _ : ''a list -> ''a expression -> bool = eval;
val test1 = eval [2, 3] (And [True, Or [Var 1, Not (Not (Var 2))],
                            Imp (Var 1, Var 2)]) = true handle NotImplemented => false;
val test2 = eval [] (And [True, Or [Var 1, Not (Not (Var 2))],
                            Imp (Var 1, Var 2)]) = false handle NotImplemented => false;
val test3 = eval [2, 3] exp1 = false handle NotImplemented => false;
val test4 = eval [] exp1 = true handle NotImplemented => false;
val test5 = eval ["A","B","D","Q"] (Eq [Var "A", Var "B",
                Imp (Var "D", Not (Var "Q")), Var "D", Var "B"]) = false handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3, test4, test5]);

val _ = print "---------- rmEmpty ----------\n";
val _ : 'a expression -> 'a expression = rmEmpty;
val test1 = rmEmpty (Or [And [Or [Eq [Not (Var 0)]]], True]) = Or [True,True] handle NotImplemented => false;
val test2 = rmEmpty
    (Eq [True, Eq [False, Var 3, And [And [], Eq [], Or [],
        Or [Var 1, Not (Eq [Var 4, False, True])], Imp (Eq [Or []], Var 2)]]]) =
    Eq [True, Eq [False, Var 3, And [True, True, False,
        Or [Var 1,Not (Eq [Var 4,False,True])], Imp (True,Var 2)]]] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2]);

val _ = print "---------- pushNegations ----------\n";
val _ : 'a expression -> 'a expression = pushNegations;
val test1 = pushNegations (Not (Imp (Not (Not (Var "a")), True))) = And [Var "a",Not True] handle NotImplemented => false;
val test2 = pushNegations
    (Not (Eq [False, Var 3, Not (And [And [], Or [Var 1, Not (Eq [])], Imp (True, Var 2)])])) =
    And [Or [Not False,Not (Var 3),And [True,Or [Var 1,Not True],Imp (True,Var 2)]],
    Or [False,Var 3,Or [Not True,And [Not (Var 1),True],And [True,Not (Var 2)]]]] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2]);

val _ = print "---------- rmConstants ----------\n";
val _ : ''a expression -> ''a expression = rmConstants;
val test1 = rmConstants (Eq [True, Var 1, Var 2]) = And [Var 1,Var 2] handle NotImplemented => false;
val test2 = rmConstants (Eq [Var 1, False, Var 2]) = And [Not (Var 1),Not (Var 2)] handle NotImplemented => false;
val test3 = rmConstants
    (Eq [Var 1, Eq [False, Var 3, And [And [], Or [Var 1, Not (Eq [Var 4, False, True])],
        Imp (True, Var 2)]]]) = Eq [Var 1,And [Not (Var 3),Not (Var 2)]] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3]);

val _ = print "---------- rmVars ----------\n";
val _ : ''a expression -> ''a expression = rmVars;
val test1 = rmVars (Or [Var "a", Var "a", Not (Var "b"), Not (Var "b")]) =
                Or [Var "a",Not (Var "b")] handle NotImplemented => false;
val test2 = rmVars (Imp (And [Var 0, Var 0] , Or [Var 0, Var 0])) = True handle NotImplemented => false;
val test3 = rmVars (Imp (And [Var 0, Var 1] , And [Var 1, Var 0])) =
                Imp (And [Var 0,Var 1],And [Var 1,Var 0]) handle NotImplemented => false;
val test4 =
    rmVars (Imp (And [Eq [Var 1, Var 0], Eq [Var 1, Var 0], Or [Var 1, Var 1]],
        And [Eq [Var 1,Var 0],Var 1])) = True handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3, test4]);

val _ = print "---------- simplify ----------\n";
val _ : ''a expression -> ''a expression = simplify;
val test1 = simplify
    (Eq [True, Eq [False, Var 3, And [And [], Or [Var 1, Not (Eq [Var 4, False, True])],
        Imp (True, Var 2)]], True,Eq [Not (Var 3), Var 2]]) = 
    And [And [Not (Var 3), Not (Var 2)], Eq [Not (Var 3), Var 2]] handle NotImplemented => false;
val test2 = simplify
    (Not (Eq [False, Var 3, Not (And [And [], Or [Var 1, Not (Eq [])], Imp (True, Var 2)])])) =
    Or [Var 3,Or [Not (Var 1),Not (Var 2)]] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2]);

val _ = print "---------- prTestEq ----------\n";
val _ : int -> ''a expression -> ''a expression -> bool = prTestEq;
val test1 = List.tabulate (20, (fn i => (i, prTestEq i exp1 exp2))) =
    [(0,true),(1,true),(2,true),(3,true),(4,true),(5,true),(6,true),(7,true),
    (8,true),(9,true),(10,true),(11,true),(12,true),(13,true),(14,true),
    (15,true),(16,true),(17,true),(18,true),(19,true)] handle NotImplemented => false;
val test2 = List.tabulate (20, (fn i => (i, prTestEq i exp1 exp3))) = 
    [(0,false),(1,true),(2,true),(3,true),(4,false),(5,false),(6,true),(7,true),
    (8,true),(9,true),(10,true),(11,true),(12,false),(13,false),(14,true),
    (15,true),(16,true),(17,true),(18,true),(19,true)] handle NotImplemented => false;
val test3 = List.tabulate (20, (fn i => (i, prTestEq i exp4 exp5))) = 
      [(0,false),(1,false),(2,true),(3,true),(4,false),(5,false),(6,true),
       (7,true),(8,false),(9,false),(10,true),(11,true),(12,false),(13,false),
       (14,true),(15,true),(16,false),(17,false),(18,true),(19,true)] handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3]);


val _ = print "---------- satSolver ----------\n";
val _ : ''a expression -> ''a list option = satSolver;
val test1 = satSolver (True : int expression) = SOME [] handle NotImplemented => false;
val test2 = satSolver (False : int expression) = NONE handle NotImplemented => false;
val test3 = satSolver (And [] : int expression) = SOME [] handle NotImplemented => false;
val test4 = satSolver (And [Or []] : int expression) = NONE handle NotImplemented => false;
val test5 = satSolver (And [Or [Var 1]]) = SOME [1] handle NotImplemented => false;
val test6 = let val sol = satSolver (And [Or [Var 1, Not (Var 2)]])
            in sol = SOME [1] orelse sol = SOME [] end handle NotImplemented => false;     
val test7 = let val sol = satSolver (And [Or [Var 1, Not (Var 1)]])
            in sol = SOME [] orelse sol = SOME [1] end handle NotImplemented => false;       
val test8 = satSolver (And [Or [Var 1, Var 3], Or [Not (Var 1), Not (Var 3)],
                        Or[Not (Var 3), Var 1]]) = SOME [1] handle NotImplemented => false;
val test9 = satSolver (And [Or [Var 1, Var 3], Or [Not (Var 1), Not (Var 3)],
                        Or[Not (Var 3), Var 1], Or[Not (Var 1), Var 3]]) = NONE handle NotImplemented => false;
val test10 = let val sol = satSolver exp6 in sol = SOME [3, 6] orelse sol = SOME [6, 3] end handle NotImplemented => false;
val _ = (all_tests := !all_tests @ [test1, test2, test3, test4, test5, test6, test6, test7, test8, test9, test10]);



(* ==================== PART 2 ==================== *)

type timetable = {day : string, time: int, course: string} list
type student = {studentID : int, curriculum : string list}

val students1 =
    [{studentID = 55, curriculum = ["DS", "P2", "RK"]},
    {studentID = 99, curriculum = ["P2", "RK", "ARS","MAFI"]}] : student list;

val students2 =
    [{studentID = 55, curriculum = ["DS", "P2", "RK"]},
    {studentID = 99, curriculum = ["P2", "RK", "ARS"]}] : student list;

val timetable =
    [{day = "cet", time = 8, course = "RK"},
    {day = "pet", time = 7, course = "DS"},
    {day = "sre", time = 10, course = "DS"},
    {day = "pet", time = 14, course = "DS"},
    {day = "sre", time = 10, course = "ARS"},
    {day = "pet", time = 14, course = "ARS"},
    {day = "tor", time = 7, course = "P2"},
    {day = "pon", time = 12, course = "P2"}] : timetable;

val timetable_tmp =
    [{day = "cetrtek", time = 8, course = "RK"},
    {day = "torek", time = 7, course = "DS"},
    {day = "sreda", time = 10, course = "DS"},
    {day = "petek", time = 14, course = "DS"},
    {day = "sreda", time = 10, course = "ARS"},
    {day = "petek", time = 14, course = "ARS"},
    {day = "torek", time = 7, course = "P2"},
    {day = "ponedeljek", time = 12, course = "P2"}] : timetable

val students_tmp =
    [{studentID = 63170000, curriculum = ["DS", "P2"]},
    {studentID = 63160000, curriculum = ["P2", "RK", "ARS"]}] : student list

val _ = print "---------- problemReduction + satSolver + solutionRepresentation ----------\n";
fun solve i tt sl = solutionRepresentation (satSolver (problemReduction i tt sl));
val _ : int -> timetable -> student list -> (student * timetable) list option = solve;

val tmp = problemReduction 2 timetable_tmp students_tmp;

val test1 = solve 1 timetable students2 = NONE handle NotImplemented => false;
val test2 = solve 2 timetable students1 = NONE handle NotImplemented => false;
val test3 = solve 3 timetable students1 = NONE handle NotImplemented => false;
(*  for a feasible solution (in order to verify it) you need to check (by hand)
    if the following statemetns hold:
        1. for each student A: A's timetable is a subset of the global timetable
        2. no student is missing in the output
        3. for each student A: A's timetable covers his curriculum
        4. for each student A: A's timetable has no overlaps
        5. for each cycle C: #students on a cycle C  < room capacity *)
val test4 = solve 2 timetable students2 handle NotImplemented => NONE;
val _ = (all_tests := !all_tests @ [test1, test2, test3]);


val nr_passes_tests = foldl (fn (true, acc) => acc + 1 | (false, acc) => acc) 0 (!all_tests);
val nr_all_tests = length (!all_tests);

val _ = if nr_passes_tests = nr_all_tests
        then OS.Process.exit OS.Process.success
        else OS.Process.exit OS.Process.failure;