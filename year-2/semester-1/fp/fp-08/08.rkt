#lang racket

;; Define data types
(struct int (n) #:transparent)
(struct true () #:transparent)
(struct false () #:transparent)
(struct add (e1 e2) #:transparent)
(struct mul (e1 e2) #:transparent)
(struct ~ (e) #:transparent)
(struct ?int (e) #:transparent)
(struct ?bool (e) #:transparent)
(struct ?leq (e1 e2) #:transparent)
(struct if-then-else (e1 e2 e3) #:transparent)

;; Helper functions to check types
(define (is-int? e) (int? e))
(define (is-bool? e) (or (true? e) (false? e)))

;; Evaluate FR expressions
(define (fri expr)
  (match expr
    [(int n) expr]
    [(true) expr]
    [(false) expr]
    [(add e1 e2)
     (let ([v1 (fri e1)] [v2 (fri e2)])
       (cond
         [(and (is-int? v1) (is-int? v2)) (int (+ (int-n v1) (int-n v2)))]
         [(and (is-bool? v1) (is-bool? v2)) (if (or (true? v1) (true? v2)) (true) (false))]
         [else (error "Invalid add operation")]))]
    [(mul e1 e2)
     (let ([v1 (fri e1)] [v2 (fri e2)])
       (cond
         [(and (is-int? v1) (is-int? v2)) (int (* (int-n v1) (int-n v2)))]
         [(and (is-bool? v1) (is-bool? v2)) (if (and (true? v1) (true? v2)) (true) (false))]
         [else (error "Invalid mul operation")]))]
    [(~ e)
     (let ([v (fri e)])
       (cond
         [(is-int? v) (int (- (int-n v)))]
         [(is-bool? v) (if (true? v) (false) (true))]
         [else (error "Invalid ~ operation")]))]
    [(?int e) (if (is-int? (fri e)) (true) (false))]
    [(?bool e) (if (is-bool? (fri e)) (true) (false))]
    [(?leq e1 e2)
     (let ([v1 (fri e1)] [v2 (fri e2)])
       (cond
         [(and (is-int? v1) (is-int? v2)) (if (<= (int-n v1) (int-n v2)) (true) (false))]
         [(and (is-bool? v1) (is-bool? v2)) (if (or (false? v1) (true? v2)) (true) (false))]
         [else (error "Invalid ?leq operation")]))]
    [(if-then-else e1 e2 e3)
     (if (true? (fri e1)) (fri e2) (fri e3))]
    [else (error "Invalid expression")]))

;; Define macros
(define (conditional . clauses)
  (if (null? clauses)
      (error "Conditional requires at least one clause")
      (let loop ([cs clauses])
        (if (null? (cdr cs))
            (car cs)
            `(if-then-else ,(car cs) ,(cadr cs) ,(loop (cddr cs)))))))

(define (?geq e1 e2) `(?leq ,e2 ,e1))

;; Tests
;; (fri (int 5))                          ; => (int 5)
;; (fri (add (int 3) (int 2)))            ; => (int 5)
;; (fri (add (false) (true)))             ; => (true)
;; (fri (mul (int 3) (int 2)))            ; => (int 6)
;; (fri (mul (false) (true)))             ; => (false)
;; (fri (?leq (int 3) (int 2)))           ; => (false)
;; (fri (?leq (false) (true)))            ; => (true)
;; (fri (~ (int 3)))                      ; => (int -3)
;; (fri (~ (false)))                      ; => (true)
;; (fri (?int (int 5)))                   ; => (true)
;; (fri (if-then-else (true) (int 5) (add (int 2) (int "a")))) ; => (int 5)

;; (conditional (true) (int -100) (mul (true) (false)) (add (int 1) (int 1)) (int 9000))
