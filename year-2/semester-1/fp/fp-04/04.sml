(* Podan seznam xs agregira z začetno vrednostjo z in funkcijo f v vrednost f (f (f z s_1) s_2) s_3) ... *)
(* Aggregates xs with an initial value z and function f and returns f (f (f z s_1) s_2) s_3) ... *)
val rec reduce = fn f => fn a => fn xs => 
    case xs of [] => a
    | h::t => reduce f (f a h) t

(* Vrne seznam, ki vsebuje kvadrate števil iz vhodnega seznama. Uporabite List.map. *)
(* Returns a list of squares of the numbers. Use List.map. *)
val squares = fn xs =>
    List.map (fn x => x * x) xs

(* Vrne seznam, ki vsebuje vsa soda števila iz vhodnega seznama. Uporabite List.filter. *)
(* Returns a list that contains only even numbers from xs. Use List.filter. *)
val onlyEven = fn xs =>
    List.filter (fn x => x mod 2 = 0) xs

(* Vrne najboljši niz glede na funkcijo f (prvi arg.). Funkcija f primerja dva niza in vrne true, če je prvi niz boljši od drugega. Uporabite List.foldl. Najboljši niz v praznem seznamu je prazen niz. *)
(* Returns the best string according to the function f (first arg.). The function f compares two strings and returns true if the first string is better than the other. Use List.foldl. The best string in an empty list is an empty string. *)
val bestString = fn f => fn xs => 
    List.foldl (fn (s1, s2) => if f (s1, s2) then s1 else s2) "" xs

(* Vrne leksikografsko največji niz. Uporabite bestString. *)
(* Returns the largest string according to alphabetical ordering. Use bestString. *)
val largestString = fn xs =>
    bestString (fn (s1, s2) => s1 > s2) xs

(* Vrne najdaljši niz. Uporabite bestString. *)
(* Returns the longest string. Use bestString. *)
val longestString = fn xs =>
    bestString (fn (s1, s2) => String.size s1 > String.size s2) xs

(* Seznam uredi naraščajoče z algoritmom quicksort. Prvi argument je funkcija za primerjanje. *)
(* Sorts the list with quicksort. First argument is a compare function. *)
(* Do not use support functions in let. Uporabite lahko anonimne funkcije ter strukture List, ListPair, Math, String in Int.*)
val rec quicksort = fn cmp => fn xs =>
    case xs of
        [] => []
      | pivot::rest =>
          let
              val (less, greater) = List.partition (fn x => cmp (x, pivot) = LESS) rest
          in
              quicksort cmp less @ (pivot :: quicksort cmp greater)
          end;

(* Vrne skalarni produkt dveh vektorjev. Uporabite List.foldl in ListPair.map. *)
(* Returns the scalar product of two vectors. Use List.foldl and ListPair.map. *)
val dot = fn xs => fn ys =>
    List.foldl (fn (a, b) => a + b) 0 (ListPair.map (fn (x, y) => x * y) (xs, ys))

(* Vrne transponirano matriko. Matrika je podana z vrstičnimi vektorji od zgoraj navzdol:
  [[1,2,3],[4,5,6],[7,8,9]] predstavlja matriko
   [ 1 2 3 ]
   [ 4 5 6 ]
   [ 7 8 9 ]
*)
(* Returns the transpose of m. The matrix m is given with row vectors from top to bottom:
  [[1,2,3],[4,5,6],[7,8,9]] represents the matrix
   [ 1 2 3 ]
   [ 4 5 6 ]
   [ 7 8 9 ]
*)
val rec transpose = fn m =>
    case m of
        [] => []
      | []::_ => []
      | _ => List.map List.hd m :: transpose (List.map List.tl m);

(* Zmnoži dve matriki. Uporabite dot in transpose. *)
(* Multiplies two matrices. Use dot and transpose. *)
val multiply = fn m1 => fn m2 =>
    List.map (fn row1 => List.map (fn row2 => dot row1 row2) (transpose m2)) m1

(* V podanem seznamu prešteje zaporedne enake elemente in vrne seznam parov (vrednost, število ponovitev). Podobno deluje UNIX-ovo orodje
   uniq -c. *)
(* Counts successive equal elements and returns a list of pairs (value, count). The unix tool uniq -c works similarly. *)
val rec group = fn xs =>
    case xs of
        [] => []
      | x::_ =>
          let
              val (same, rest) = List.partition (fn y => y = x) xs
          in
              (x, List.length same) :: group rest
          end;

(* Elemente iz podanega seznama razvrsti v ekvivalenčne razrede. Znotraj razredov naj bodo elementi v istem vrstnem redu kot v podanem seznamu. Ekvivalentnost elementov definira funkcija f, ki za dva elementa vrne true, če sta ekvivalentna. *)
(* Sorts the elements from a list into equivalence classes. The order of elements inside each equivalence class should be the same as in the original list. The equivalence relation is given with a function f, which returns true, if two elements are equivalent. *)
val rec equivalenceClasses = fn f => fn xs =>
    case xs of
        [] => []
      | x::_ =>
          let
              val (same, rest) = List.partition (fn y => f x y) xs
          in
              same :: equivalenceClasses f rest
          end;
