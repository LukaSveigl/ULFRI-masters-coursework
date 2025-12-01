#lang racket

(provide false true int .. empty exception
         trigger triggered handle
         if-then-else
         ?int ?bool ?.. ?seq ?empty ?exception
         add mul ?leq ?= head tail ~ ?all ?any
         vars valof fun proc closure call
         greater rev binary filtering folding mapping
         fri)

;;; The FRI programming language constructs.
;;; - Data type definitions.
(struct false () #:transparent)
(struct true () #:transparent)
(struct int (n) #:transparent)
(struct .. (e1 e2) #:transparent)
(struct empty () #:transparent)
(struct exception (exn) #:transparent)
(struct triggered (exn) #:transparent)

;;; - Type-checking structures.
(struct ?int (e) #:transparent)
(struct ?bool (e) #:transparent)
(struct ?.. (e) #:transparent)
(struct ?seq (e) #:transparent)
(struct ?empty (e) #:transparent)
(struct ?exception (e) #:transparent)

;;; - Flow control structures.
(struct trigger (e) #:transparent)
(struct handle (e1 e2 e3) #:transparent)
(struct if-then-else (condition e1 e2) #:transparent)
(struct add (e1 e2) #:transparent)
(struct mul (e1 e2) #:transparent)
(struct ?leq (e1 e2) #:transparent)
(struct ?= (e1 e2) #:transparent)
(struct head (e) #:transparent)
(struct tail (e) #:transparent)
(struct ~ (e) #:transparent)
(struct ?all (e) #:transparent)
(struct ?any (e) #:transparent)

;;; - Variable structures.
(struct vars (s e1 e2) #:transparent)
(struct valof (s) #:transparent)

;;; - Function and procedure structures.
(struct fun (name fargs body) #:transparent)
(struct proc (name body) #:transparent)
(struct closure (env f) #:transparent)
(struct call (e args) #:transparent)


;;; Implementations of FRI language constructs.
;;; - Type-checking structures.

;; Checks if the given expression is an integer type. Returns true if it is, false otherwise, 
;; propagating triggered exceptions.
(define (?int-impl e env)
    (let ([result-e (fri e env)])
        (if (triggered? result-e)
            result-e
            (if (int? result-e) (true) (false)))))

;; Checks if the given expression is a boolean type. Returns true if it is, false otherwise,
;; propagating triggered exceptions.
(define (?bool-impl e env)
    (let ([result-e (fri e env)])
        (if (triggered? result-e)
            result-e
            (if (or (true? result-e) (false? result-e)) (true) (false)))))

;; Checks if the given expression is a sequence type. Returns true if it is, false otherwise,
;; propagating triggered exceptions.
(define (?..-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (..? result-e) (true) (false)))))

;; Checks if the given expression is a sequence type. Returns true if it is, false otherwise,
;; propagating triggered exceptions.
(define (?seq-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (or (empty? result-e ) (..? result-e))
            (if (empty? result-e) (true) (?seq-impl (..-e2 result-e) env))
            (false)))))

;; Checks if the given expression is an empty sequence. Returns true if it is, false otherwise,
;; propagating triggered exceptions.
(define (?empty-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (empty? result-e) (true) (false)))))

;; Checks if the given expression is an exception. Returns true if it is, false otherwise,
;; propagating triggered exceptions.
(define (?exception-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (exception? result-e) (true) (false)))))


;;; - Flow control structures.

;; Triggers the given expression, propagating triggered exceptions.
(define (trigger-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (?exception result-e)
            (triggered result-e)
            (triggered (exception "trigger: wrong argument type"))))))

;; Handles the given expression, propagating triggered exceptions.
(define (handle-impl e1 e2 e3 env)
  (let ([result-e1 (fri e1 env)]
        [result-e2 (fri e2 env)]
        [result-e3 (fri e3 env)])
    (cond
      [(triggered? result-e1) result-e1]
      [(not (true?(?exception-impl result-e1 env))) (triggered (exception "handle: wrong argument type"))]
      [else (if (and (triggered? result-e2)
                     (string=? (exception-exn (triggered-exn result-e2)) (exception-exn result-e1)))
              result-e3
              result-e2)])))

;; Evaluates the given condition, and returns the result of the first expression if the condition is true,
;; otherwise returns the result of the second expression, propagating triggered exceptions.
(define (if-then-else-impl condition e1 e2 env)
  (let ([result-condition (fri condition env)])
    (if (triggered? result-condition)
        result-condition
        (if (false? result-condition) (fri e2 env) (fri e1 env)))))

;; Adds two expressions together, propagating triggered exceptions.
;; If both expressions are integers, returns the sum of the two integers.
;; If both expressions are sequences, returns the concatenation of the two sequences.
;; If both expressions are booleans, returns the logical OR of the two booleans.
(define (add-impl e1 e2 env)
  (let ([result-e1 (fri e1 env)]
        [result-e2 (fri e2 env)])
    (cond
      [(triggered? result-e1) result-e1]
      [(triggered? result-e2) result-e2]
      [(and (true?(?bool-impl result-e1 env)) (true?(?bool-impl result-e2 env))) (if (or (true? result-e1) (true? result-e2)) (true) (false))]
      [(and (true?(?int-impl result-e1 env)) (true?(?int-impl result-e2 env))) (int (+ (int-n result-e1) (int-n result-e2)))]
      [(and (true?(?seq-impl result-e1 env)) (true?(?seq-impl result-e2 env)))
       (if (true?(?empty-impl result-e1 env))
           result-e2
           (.. (..-e1 result-e1) (add-impl (..-e2 result-e1) result-e2 env)))]
      [else (triggered (exception "add: wrong argument type"))])))

;; Multiplies two expressions together, propagating triggered exceptions.
;; If both expressions are integers, returns the product of the two integers.
;; If both expressions are booleans, returns the logical AND of the two booleans.
(define (mul-impl e1 e2 env)
  (let ([result-e1 (fri e1 env)]
        [result-e2 (fri e2 env)])
    (cond
      [(triggered? result-e1) result-e1]
      [(triggered? result-e2) result-e2]
      [(and (true?(?bool-impl result-e1 env)) (true?(?bool-impl result-e2 env))) (if (and (true? result-e1) (true? result-e2)) (true) (false))]
      [(and (true?(?int-impl result-e1 env)) (true?(?int-impl result-e2 env))) (int (* (int-n result-e1) (int-n result-e2)))]
      [else (triggered (exception "mul: wrong argument type"))])))

;; Checks if the first expression is less than or equal to the second expression, propagating triggered exceptions.
(define (?leq-impl e1 e2 env)
  (let ([result-e1 (fri e1 env)]
        [result-e2 (fri e2 env)])
    (cond
      [(triggered? result-e1) result-e1]
      [(triggered? result-e2) result-e2]
      [(and (true?(?bool-impl result-e1 env)) (true?(?bool-impl result-e2 env))) (if (or (false? result-e1) (true? result-e2)) (true) (false))]
      [(and (true?(?int-impl result-e1 env)) (true?(?int-impl result-e2 env))) (if (<= (int-n result-e1) (int-n result-e2)) (true) (false))]
      [(and (true?(?seq-impl result-e1 env)) (true?(?seq-impl result-e2 env))) (if (<= (length (seq->list result-e1)) (length (seq->list result-e2))) (true) (false))]
      [else (triggered (exception "?leq: wrong argument type"))])))

;; Checks if the two expressions are equal, propagating triggered exceptions.
(define (?=-impl e1 e2 env)
  (let ([result-e1 (fri e1 env)]
        [result-e2 (fri e2 env)])
    (if (triggered? result-e1)
        result-e1
        (if (triggered? result-e2)
            result-e2
            (if (equal? result-e1 result-e2) (true) (false))))))

;; Returns the head of the given sequence, propagating triggered exceptions.
(define (head-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (?seq-impl result-e env)
            (if (empty? result-e)
                (triggered (exception "head: empty sequence"))
                (if (..? result-e)
                    (..-e1 result-e)
                    (triggered (exception "head: wrong argument type"))))
            (triggered (exception "head: wrong argument type"))))))

;; Returns the tail of the given sequence, propagating triggered exceptions.
(define (tail-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (?seq-impl result-e env)
            (if (empty? result-e)
                (triggered (exception "tail: empty sequence"))
                (if (..? result-e)
                    (..-e2 result-e)
                    (triggered (exception "tail: wrong argument type"))))
            (triggered (exception "tail: wrong argument type"))))))

;; Returns the negation of the given expression, propagating triggered exceptions.
;; If the expression is a boolean, returns the logical negation of the boolean.
;; If the expression is an integer, returns the negation of the integer.
(define (~-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (cond
          [(true?(?bool-impl result-e env)) (if (true? result-e) (false) (true))]
          [(true?(?int-impl result-e env)) (int (- (int-n result-e)))]
          [else (triggered (exception "~: wrong argument type"))]))))

;; Checks if all elements in the given sequence are true, propagating triggered exceptions.
(define (?all-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (true?(?seq-impl result-e env))
            (let loop ([seq result-e])
              (if (empty? seq)
                  (true)
                  (let ([head-val (fri (..-e1 seq) env)])
                    (if (triggered? head-val)
                        head-val
                        (if (false? head-val)
                            (false)
                            (loop (..-e2 seq)))))))
            (triggered (exception "?all: wrong argument type"))))))

;; Checks if any element in the given sequence is true, propagating triggered exceptions.
(define (?any-impl e env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (if (true?(?seq-impl result-e env))
            (let loop ([seq result-e])
              (if (empty? seq)
                  (false)
                  (let ([head-val (fri (..-e1 seq) env)])
                    (if (triggered? head-val)
                        head-val
                        (if (false? head-val)
                            (loop (..-e2 seq))
                            (true))))))
            (triggered (exception "?any: wrong argument type"))))))


;;; - Variable structures.

;; Binds the given variable to the given expression in the environment, propagating triggered exceptions.
(define (vars-impl s e1 e2 env)
  (let ([new-env (if (list? s)
                     (let loop ([s s] [e1 e1] [acc-env '()])
                       (if (null? s)
                           (append (reverse acc-env) env)
                           (let ([result (fri (car e1) env)])
                             (if (triggered? result)
                                 (triggered (exception (exception-exn (triggered-exn result))))
                                 (loop (cdr s) (cdr e1) (cons (cons (car s) result) acc-env)))))) 
                     (let ([result (fri e1 env)])
                       (if (triggered? result)
                           (triggered (exception (exception-exn (triggered-exn result))))
                           (cons (cons s result) env))))])
    (if (triggered? new-env)
        new-env
        (fri e2 new-env))))

;; Returns the value of the given variable, propagating triggered exceptions.
;; If the variable is not defined in the environment, triggers an exception.
(define (valof-impl s env)
  (let ([result (assoc s env)])
    (if result
        (cdr result)
        (triggered (exception "valof: undefined variable")))))


;;; - Function and procedure structures.

;; Creates a closure with the given environment and function, propagating triggered exceptions.
;; It is used to evaluate functions with lexical scoping. Additionally, it checks for undefined variables
;; in the function body.
(define (fun-impl name fargs body env)
  (define (check-undefined-vars expr env)
    (match expr
      [(valof s)
       (if (assoc s env)
           #f
           (triggered (exception "closure: undefined variable")))]
      [(vars s e1 e2)
       (let ([new-env (if (list? s)
                          (append (map (lambda (x y) (cons x (fri y env))) s e1) env)
                          (cons (cons s (fri e1 env)) env))])
         (or (check-undefined-vars e1 env)
             (check-undefined-vars e2 new-env)))]
      [(fun name fargs body)
       (let ([new-env (append (map (lambda (x) (cons x #f)) fargs) env)])
         (check-undefined-vars body new-env))]
      [(proc name body)
       (check-undefined-vars body env)]
      [(call e args)
       (or (check-undefined-vars e env)
           (ormap (lambda (arg) (check-undefined-vars arg env)) args))]
      [(add e1 e2)
       (or (check-undefined-vars e1 env)
           (check-undefined-vars e2 env))]
      [(mul e1 e2)
       (or (check-undefined-vars e1 env)
           (check-undefined-vars e2 env))]
      [(list e1 e2)
       (or (check-undefined-vars e1 env)
           (check-undefined-vars e2 env))]
      [else #f]))

  (if (not (equal? (length fargs) (length (remove-duplicates fargs))))
      (triggered (exception "fun: duplicate argument identifier"))
      (let ([undefined-var (check-undefined-vars body (append (map (lambda (x) (cons x #f)) fargs)
                                                             (if (string=? name "") env (cons (cons name #f) env))))])
        (if undefined-var
            undefined-var
            (closure env (fun name fargs body))))))

;; Creates a procedure with the given environment and body
(define (proc-impl name body env)
  (proc name body))

;; Calls the given expression (function/procedure) with the given arguments, propagating triggered exceptions.
(define (call-impl e args env)
  (let ([result-e (fri e env)])
    (if (triggered? result-e)
        result-e
        (match result-e
          ; Function evaluates to a closure - uses lexical scoping to evaluate the function body.
          [(closure closure-env (fun name fargs body))
           (if (not (= (length fargs) (length args)))
               (triggered (exception "call: wrong number of arguments"))
               (let ([new-env (append (map (lambda (x y) (cons x (fri y env))) fargs args)
                                      (if (string=? name "") closure-env (cons (cons name result-e) closure-env)))])
                 (fri body new-env)))]
          ; Procedure evaluates to a procedure - uses dynamic scoping to evaluate the procedure body.
          [(proc name body)
           (if (null? args)
               (fri body (cons (cons name result-e) env))
               (triggered (exception "call: arity mismatch")))]
          [else (triggered (exception "call: wrong argument type"))]))))


;;; Macros - written in the FRI language.
;;; - Macros for greater, rev, binary, filtering, folding, and mapping.

;;; Greater - checks if the first expression is greater than the second expression.
(define (greater e1 e2)
  (if-then-else (?leq e2 e1)
                (if-then-else (?= e1 e2) (false) (true))
                (false)))


;;; Rev - reverses the given sequence.
;; Example: (fri (rev (.. (int 1) (empty))) null) => (.. (int 1) (empty))
;; Example: (fri (rev (.. (int 1) (.. (int 2) (.. (int 3) (.. (int 4) (empty)))))) null) => (.. (int 4) (.. (int 3) (.. (int 2) (.. (int 1) (empty))))
;; Uses only the FRI language constructs.
(define (rev e)
    (call (fun "rev" (list "e" "acc")
        (if-then-else (?empty (valof "e"))
            (valof "acc")
            (call (valof "rev") 
                (list (tail (valof "e")) 
                    (.. (head (valof "e")) (valof "acc"))))))
        (list e (empty))))

;;; Binary - If the result of e1 is a positive integer, returns a sequence of bits representing the integer.
;;; The binary macro defines the function binary, that defines a function mod which defines a function div.
;;; The functions binary, mod and div are implemented using the FRI language constructs.
(define (binary e1) 
    (rev 
        (call (fun "binary" (list "e1")
            (if-then-else (?leq (valof "e1") (int 0))
              (empty)    
              (.. (call (fun "mod" (list "x")
                          (if-then-else (?leq (valof "x") (int 1))
                            (valof "x")
                            (call (valof "mod") (list (add (valof "x") (~ (int 2)))))))
                    (list (valof "e1")))
                  (call (valof "binary") 
                    (list (call (fun "div" (list "x" "acc")
                                (if-then-else (?leq (valof "x") (int 1))
                                  (valof "acc")
                                  (call (valof "div") 
                                    (list (add (valof "x") (~ (int 2))) (add (valof "acc") (int 1))))))
                          (list (valof "e1") (int 0))))))))
            (list e1))))

;;; Mapping - A macro that generates code that applies the given function expression to each 
;;; element in the sequence. Similar to List.map in SML, but without currying. f is an expression, that
;;; evaluates into a functional closure, and seq is the expression, that evaluates into a sequence of elements.
(define (mapping f seq)
    (call (fun "mapping" (list "f" "seq")
        (if-then-else (?empty (valof "seq"))
            (empty)
            (.. (call (valof "f") (list (head (valof "seq"))))
                (call (valof "mapping") (list (valof "f") (tail (valof "seq")))))))
        (list f seq)))

;;; Filtering - A macro that generates code that filters the given sequence based on the given function expression.
;;; The function expression evaluates into a functional closure, and seq is an expression, that evaluates into
;;; a sequence of elements. Similar to List.filter in SML, but without currying.
(define (filtering f seq)
    (call (fun "filtering" (list "f" "seq")
        (if-then-else (?empty (valof "seq"))
            (empty)
            (if-then-else (call (valof "f") (list (head (valof "seq"))))
                (.. (head (valof "seq"))
                    (call (valof "filtering") (list (valof "f") (tail (valof "seq"))))
                )
                (call (valof "filtering") (list (valof "f") (tail (valof "seq")))))))
        (list f seq)))

;;; Filtering - A macro that generates code that folds the given sequence based on the given function expression.
;;; The function expression evaluates into a functional closure, and seq is an expression, that evaluates into
;;; a sequence of elements. Similar to List.fold in SML, but without currying.
(define (folding f init seq) 
    (call (fun "folding" (list "f" "init" "seq")
            (if-then-else (?empty (valof "seq"))
                (valof "init")
                (call (valof "folding") 
                    (list (valof "f") 
                        (call (valof "f") (list (head (valof "seq")) (valof "init")))
                        (tail (valof "seq"))))))
        (list f init seq)))


;;; Un-implemented functions.
(define (not-implemented)
  (triggered (exception "not implemented")))

;;; FRI - interpreter for the language. Triggers a recursive descent evaluation of the given expression,
;;; passing the environment along the way.
(define (fri expr env)
  (match expr

    ;; Constants.
    [(false) (false)]
    [(true) (true)]
    [(int n) (int n)]
    [(.. e1 e2) 
     (let ([result-e1 (fri e1 env)]
           [result-e2 (fri e2 env)])
       (if (triggered? result-e1)
           result-e1
           (if (triggered? result-e2)
               result-e2
               (.. result-e1 result-e2))))]
    [(empty) (empty)]
    [(exception exn) (exception exn)]
    [(triggered exn) (triggered exn)]

    ;; Type-checking.
    [(?int e) (?int-impl (fri e env) env)]
    [(?bool e) (?bool-impl (fri e env) env)]
    [(?.. e) (?..-impl (fri e env) env)]
    [(?seq e) (?seq-impl (fri e env) env)]
    [(?empty e) (?empty-impl (fri e env) env)]
    [(?exception e) (?exception-impl (fri e env) env)]

    ;; Flow control.
    [(trigger e) (trigger-impl e env)]
    [(handle e1 e2 e3) (handle-impl e1 e2 e3 env)]
    [(if-then-else condition e1 e2) (if-then-else-impl condition e1 e2 env)]
    [(add e1 e2) (add-impl e1 e2 env)]
    [(mul e1 e2) (mul-impl e1 e2 env)]
    [(?leq e1 e2) (?leq-impl e1 e2 env)]
    [(?= e1 e2) (?=-impl e1 e2 env)]
    [(head e) (head-impl e env)]
    [(tail e) (tail-impl e env)]
    [(~ e) (~-impl e env)]
    [(?all e) (?all-impl e env)]
    [(?any e) (?any-impl e env)]

    ;; Variables.
    [(vars s e1 e2) (vars-impl s e1 e2 env)]
    [(valof s) (valof-impl s env)]

    ;; Functions and procedures.
    [(fun name fargs body) (fun-impl name fargs body env)]
    [(proc name body) (proc-impl name body env)]
    [(call e args) (call-impl e args env)]

    ;;[else (triggered (exception "fri: unknown expression type"))]))
    ;; Add the expression type to the triggered exception.
    [else (triggered (exception (format "fri: unknown expression type: ~a" expr)))]))

;; Helper function to convert sequence to list.
(define (seq->list seq)
  (if (empty? seq)
      '()
      (cons (..-e1 seq) (seq->list (..-e2 seq)))))