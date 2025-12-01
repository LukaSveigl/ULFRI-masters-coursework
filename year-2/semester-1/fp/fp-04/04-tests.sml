use "04.sml";

Control.printWarnings := false;

(* Display longer expressions in interpreter output *)
val _ = Control.Print.printDepth := 100;
val _ = Control.Print.printLength := 1000;
val _ = Control.Print.stringDepth := 1000;

val _ = print "~~~~~~~~ reduce ~~~~~~~~\n";
val _ : (('a -> 'b -> 'a) -> 'a -> 'b list -> 'a) = reduce;
val reduce_test1 = reduce (fn a => fn b => a + b) 0 [1, 2, 3, 4, 5] = 15;
val reduce_test2 = reduce (fn a => fn b => a * b) 1 [1, 2, 3, 4, 5] = 120;
val reduce_test3 = reduce (fn a => fn b => a - b) 10 [1, 2, 3] = 4;

val _ = print "~~~~~~~~ squares ~~~~~~~~\n";
val _ : (int list -> int list) = squares;
val squares_test1 = (squares [1, 2, 3, 4] = [1, 4, 9, 16]);
val squares_test2 = (squares [] = []);

val _ = print "~~~~~~~~ onlyEven ~~~~~~~~\n";
val _ : (int list -> int list) = onlyEven;
val onlyEven_test1 = (onlyEven [1, 2, 3, 4, 5, 6] = [2, 4, 6]);
val onlyEven_test2 = (onlyEven [1, 3, 5] = []);
val onlyEven_test3 = (onlyEven [] = []);

val _ = print "~~~~~~~~ bestString ~~~~~~~~\n";
val _ : ((string * string -> bool) -> string list -> string) = bestString;
val bestString_test1 = (bestString (fn (s1, s2) => s1 > s2) ["apple", "banana", "grape"] = "grape");
val bestString_test2 = (bestString (fn (s1, s2) => String.size s1 > String.size s2) ["a", "ab", "abc"] = "abc");
val bestString_test3 = (bestString (fn (_, _) => true) [] = "");

val _ = print "~~~~~~~~ largestString ~~~~~~~~\n";
val largestString_test1 = (largestString ["apple", "banana", "grape"] = "grape");
val largestString_test2 = (largestString [] = "");

val _ = print "~~~~~~~~ longestString ~~~~~~~~\n";
val longestString_test1 = (longestString ["apple", "banana", "cherry"] = "banana");
val longestString_test2 = (longestString [] = "");

val _ = print "~~~~~~~~ quicksort ~~~~~~~~\n";
val quicksort_test1 = (quicksort (fn (x, y) => Int.compare (x, y)) [3, 1, 4, 1, 5, 9] = [1, 1, 3, 4, 5, 9]);
val quicksort_test2 = (quicksort (fn (x, y) => Int.compare (y, x)) [3, 1, 4, 1, 5, 9] = [9, 5, 4, 3, 1, 1]);
val quicksort_test3 = (quicksort (fn (x, y) => Int.compare (x, y)) [] = []);

val _ = print "~~~~~~~~ dot ~~~~~~~~\n";
val dot_test1 = (dot [1, 2, 3] [4, 5, 6] = 32);
val dot_test2 = (dot [1, 2, 3] [0, 0, 0] = 0);
val dot_test3 = (dot [] [] = 0);

val _ = print "~~~~~~~~ transpose ~~~~~~~~\n";
val transpose_test1 = (transpose [[1,2,3],[4,5,6],[7,8,9]] = [[1,4,7],[2,5,8],[3,6,9]]);
val transpose_test2 = (transpose [[1,2,3]] = [[1],[2],[3]]);
val transpose_test3 = (transpose [] = []);

val _ = print "~~~~~~~~ multiply ~~~~~~~~\n";
val multiply_test1 = (multiply [[1,2],[3,4]] [[5,6],[7,8]] = [[19,22],[43,50]]);
val multiply_test2 = (multiply [[1,0],[0,1]] [[1,2],[3,4]] = [[1,2],[3,4]]);
val multiply_test3 = (multiply [[0]] [[0]] = [[0]]);

val _ = print "~~~~~~~~ group ~~~~~~~~\n";
val group_test1 = (group [1,1,2,2,2,3,3] = [(1,2), (2,3), (3,2)]);
val group_test2 = (group [1,2,3] = [(1,1), (2,1), (3,1)]);
val group_test3 = (group [] = []);

val _ = print "~~~~~~~~ equivalenceClasses ~~~~~~~~\n";
val _ : ('a -> 'a -> bool) -> 'a list -> 'a list list = equivalenceClasses;
val equivalenceClasses_test1 = equivalenceClasses (fn a => fn b => a = b) [1, 2, 2, 3, 3, 3, 4, 4, 4, 4] = [[1], [2, 2], [3, 3, 3], [4, 4, 4, 4]];
val equivalenceClasses_test2 = equivalenceClasses (fn a => fn b => a = b) ["a", "b", "b", "c", "c", "c"] = [["a"], ["b", "b"], ["c", "c", "c"]];
val equivalenceClasses_test3 = equivalenceClasses (fn a => fn b => a = b) [1, 2, 3, 4, 5] = [[1], [2], [3], [4], [5]];
val equivalenceClasses_test4 = equivalenceClasses (fn a => fn b => a = b) [] = [];
val equivalenceClasses_test5 = equivalenceClasses (fn a => fn b => a = b) [1] = [[1]];
val equivalenceClasses_test6 = equivalenceClasses (fn a => fn b => (a mod 2) = (b mod 2)) [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] = [[1, 3, 5, 7, 9], [2, 4, 6, 8, 10]];

(*
(* Print all test results *)
val _ = print "~~~~~~~~ Test Results ~~~~~~~~\n";
val _ = print (if reduce_test1 andalso reduce_test2 andalso reduce_test3 then "reduce tests passed\n" else "reduce tests failed\n");
val _ = print (if squares_test1 andalso squares_test2 then "squares tests passed\n" else "squares tests failed\n");
val _ = print (if onlyEven_test1 andalso onlyEven_test2 andalso onlyEven_test3 then "onlyEven tests passed\n" else "onlyEven tests failed\n");
val _ = print (if bestString_test1 andalso bestString_test2 andalso bestString_test3 then "bestString tests passed\n" else "bestString tests failed\n");
val _ = print (if largestString_test1 andalso largestString_test2 then "largestString tests passed\n" else "largestString tests failed\n");
val _ = print (if longestString_test1 andalso longestString_test2 then "longestString tests passed\n" else "longestString tests failed\n");
val _ = print (if quicksort_test1 andalso quicksort_test2 andalso quicksort_test3 then "quicksort tests passed\n" else "quicksort tests failed\n");
val _ = print (if dot_test1 andalso dot_test2 andalso dot_test3 then "dot tests passed\n" else "dot tests failed\n");
val _ = print (if transpose_test1 andalso transpose_test2 andalso transpose_test3 then "transpose tests passed\n" else "transpose tests failed\n");
val _ = print (if multiply_test1 andalso multiply_test2 andalso multiply_test3 then "multiply tests passed\n" else "multiply tests failed\n");
val _ = print (if group_test1 andalso group_test2 andalso group_test3 then "group tests passed\n" else "group tests failed\n");
val _ = print (if equivalenceClasses_test1 andalso equivalenceClasses_test2 andalso equivalenceClasses_test3 then "equivalenceClasses tests passed\n" else "equivalenceClasses tests failed\n");
*)