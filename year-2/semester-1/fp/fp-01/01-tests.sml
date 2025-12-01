use "01.sml";

val _ = print "~~~~~~~~ factorial ~~~~~~~~\n";
val test_type: int -> int = factorial;
val test = factorial 5 = 120;

val _ = print "~~~~~~~~ power ~~~~~~~~\n";
val test_type: int * int -> int = power;
val test = power (2, 3) = 8;

val _ = print "~~~~~~~~ gcd ~~~~~~~~\n";
val test_type: int * int -> int = gcd;
val test = gcd (54, 24) = 6;

val _ = print "~~~~~~~~ len ~~~~~~~~\n";
val test_type: int list -> int = len;
val test = len [1, 2, 3, 4, 5] = 5;

val _ = print "~~~~~~~~ last ~~~~~~~~\n";
val test_type: int list -> int option = last;
val test = last [1, 2, 3, 4, 5] = SOME 5;

val _ = print "~~~~~~~~ nth ~~~~~~~~\n";
val test_type: int list * int -> int option = nth;
val test = nth ([1, 2, 3, 4, 5], 2) = SOME 3;

val _ = print "~~~~~~~~ insert ~~~~~~~~\n";
val test_type: int list * int * int -> int list = insert;
val test = insert ([1, 2, 3, 4, 5], 2, 99) = [1, 2, 99, 3, 4, 5];

val _ = print "~~~~~~~~ delete ~~~~~~~~\n";
val test_type: int list * int -> int list = delete;
val test = delete ([1, 2, 3, 4, 5, 3], 3) = [1, 2, 4, 5];

val _ = print "~~~~~~~~ reverse ~~~~~~~~\n";
val test_type: int list -> int list = reverse;
val test = reverse [1, 2, 3, 4, 5] = [5, 4, 3, 2, 1];

val _ = print "~~~~~~~~ palindrome ~~~~~~~~\n";
val test_type: int list -> bool = palindrome;
val test = palindrome [1, 2, 3, 2, 1] = true;