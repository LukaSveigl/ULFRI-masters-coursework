
(*
	DO NOT USE THIS CODE IN YOUR PROJECT.
	YOU CAN ONLY USE IT FOR TESTING PURPOSES.
	THE CODE IS INTENTIONALY BADLY WRITTEN.
    WINDOWS USERS: make sure you have "cryptominisat5-win-amd64-nogauss.exe" in the working directory
    LINUX USERS: make sure that "./cryptominisat5_amd64_linux" is executable
*)

fun external_sat_solver (And es) =
let
	fun vars1 es = 
	List.foldl (fn (e, es) => e :: (List.filter (fn x => e <> x) es)) []
	(List.concat (List.map (fn Or es => List.map 
		(fn (Not (Var x) | Var x) => x | _ => raise NotImplemented)
		es | _ => raise NotImplemented) es))

    val vars = vars1 es

    fun lookup _ [] index = index
    |   lookup x (e :: es) index = if x = e then index else lookup x es (index + 1)

    val (clauses : int list list) = List.map
        (fn Or es => 
            (List.map
                (fn Var v => lookup v vars 1
                |   Not (Var v) => ~(lookup v vars 1)
                |   _ => raise NotImplemented) es)
        |   _ => raise NotImplemented) es

    fun writeDimacs nv nc cls =
		let
			val os = TextIO.openOut "tempSAT"
			fun string_of_int i = if i < 0 then "-" ^ Int.toString (~i) else Int.toString i
		in
			TextIO.output (os,
			"p cnf " ^ Int.toString nv ^ " " ^ Int.toString nc ^ "\n"
			^ String.concat (List.map (fn c => String.concatWith " " (List.map string_of_int c) ^ " 0\n") cls));
		TextIO.closeOut os
	end;

	fun readDimacs () =
	let
		val is = TextIO.openIn "tempSOL"
		fun read_lines is =
			case TextIO.inputLine is of
				SOME line =>
					(if String.size line > 1 andalso String.sub (line, 0) = #"v"
					then List.map (fn s => case Int.fromString s of SOME i => i | NONE => raise NotImplemented)
						(String.tokens (fn x => x = #" ")
							(String.map (fn #"-" => #"~" | (#"v" | #"\n") => #" " | c => c) line))
					else []) :: read_lines is
			|	NONE => [] 
	in
		List.concat (read_lines is) before TextIO.closeIn is
	end;

    fun solve cls =
	let
    	val _ = writeDimacs (List.length vars) (List.length cls) cls
		val _ = if String.isPrefix "/" (OS.FileSys.getDir ())
    			then ignore (OS.Process.system "./cryptominisat5_amd64_linux tempSAT > tempSOL")	
				else ignore (OS.Process.system "cryptominisat5-win-amd64-nogauss.exe tempSAT > tempSOL")
    	val sol = readDimacs ()
	in
		if sol = []
		then NONE
		else SOME (List.filter (fn x => x > 0) sol)
	end
in
	case solve clauses of
        NONE => NONE
    |   SOME setVars => SOME (map (fn i => List.nth (vars, i - 1)) setVars)
end
|	external_sat_solver _ = raise NotImplemented;
