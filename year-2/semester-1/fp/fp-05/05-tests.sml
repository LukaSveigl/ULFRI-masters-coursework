use "05.sml";

val _ = print "~~~~~~~~~~~~~~~~~Rational~~~~~~~~~~~~~~~~~\n";
val test1 = Rational.makeRational (1, 2) = Rational.Frac (1, 2);
val test2 = Rational.makeRational (1, 1) = Rational.Whole 1;
val test3 = Rational.makeRational (0, 1) = Rational.Whole 0;
val test4 = Rational.makeRational (1, ~2) = Rational.Frac (~1, 2);
val test5 = Rational.makeRational (1, ~1) = Rational.Whole (~1);

val test6 = Rational.neg (Rational.makeRational (1, 2)) = Rational.Frac (~1, 2);
val test7 = Rational.neg (Rational.makeRational (1, 1)) = Rational.Whole (~1);
val test8 = Rational.neg (Rational.makeRational (0, 1)) = Rational.Whole 0;
val test9 = Rational.neg (Rational.makeRational (1, ~2)) = Rational.Frac (1, 2);
val test10 = Rational.neg (Rational.makeRational (1, ~1)) = Rational.Whole 1;

val test11 = Rational.inv (Rational.makeRational (1, 2)) = Rational.makeRational (2, 1);
val test12 = Rational.inv (Rational.makeRational (1, 1)) = Rational.makeRational (1, 1);
val test14 = Rational.inv (Rational.makeRational (1, ~2)) = Rational.makeRational (2, ~1);
val test15 = Rational.inv (Rational.makeRational (1, ~1)) = Rational.makeRational (~1, 1);

val test16 = Rational.add (Rational.makeRational (1, 2), Rational.makeRational (1, 2)) = 
    Rational.makeRational (1, 1);
val test17 = Rational.add (Rational.makeRational (1, 1), Rational.makeRational (1, 2)) =
    Rational.makeRational (3, 2);
val test18 = Rational.add (Rational.makeRational (1, 2), Rational.makeRational (1, 1)) =
    Rational.makeRational (3, 2);
val test19 = Rational.add (Rational.makeRational (1, 2), Rational.makeRational (1, 1)) =
    Rational.makeRational (3, 2);
val test20 = Rational.add (Rational.makeRational (1, 2), Rational.makeRational (2, 2)) =
    Rational.makeRational (3, 2);

val test21 = Rational.mul (Rational.makeRational (1, 2), Rational.makeRational (1, 2)) =
    Rational.makeRational (1, 4);
val test22 = Rational.mul (Rational.makeRational (1, 1), Rational.makeRational (1, 2)) =
    Rational.makeRational (1, 2);
val test23 = Rational.mul (Rational.makeRational (1, 2), Rational.makeRational (1, 1)) =
    Rational.makeRational (1, 2);

val test24 = Rational.toString (Rational.makeRational (1, 2)) = "1/2";
val test25 = Rational.toString (Rational.makeRational (1, 1)) = "1";
val test26 = Rational.toString (Rational.makeRational (0, 1)) = "0";
val test27 = Rational.toString (Rational.makeRational (1, ~2)) = "~1/2";
val test28 = Rational.toString (Rational.makeRational (1, ~1)) = "~1";
val test29 = Rational.toString (Rational.makeRational (10, 2)) = "5";
val test30 = Rational.toString (Rational.makeRational (10, 1)) = "10";
val test31 = Rational.toString (Rational.makeRational (6, 4)) = "3/2";