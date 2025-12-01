#lang racket

;; Accepts a number x and a number n, and returns x^n.
;; power : Number Number -> Number
(define power (lambda (x n)
  (if (= n 0)
      1
      (* x (power x (- n 1))))))

;; Accepts two numbers and returns their greatest common divisor.
;; gcd : Number Number -> Number
(define gcd (lambda (a b)
  (if (= b 0)
      a
      (gcd b (remainder a b)))))

;; Accepts a number n and returns the n-th Fibonacci number.
;; fib : Number -> Number
(define fib (lambda (n)
    (if (<= n 1)
        n
        (+ (fib (- n 1)) (fib (- n 2))))))

;; Accepts a list and returns the reverse of the list.
;; reverse : List -> List
(define reverse (lambda (lst)
  (if (null? lst)
      '()
      (append (reverse (cdr lst)) (list (car lst))))))

;; Accepts a number and a list, and removes all occurrences of the number from the list.
;; remove-all : Number List -> List
(define remove (lambda (x lst)
  (if (null? lst)
      '()
      (if (= x (car lst))
          (remove x (cdr lst))
          (cons (car lst) (remove x (cdr lst)))))))

;; Accepts a function and a list of numbers, and applies the function to each number in the list.
;; map : (Number -> Number) List -> List
(define map (lambda (f lst)
  (if (null? lst)
      '()
      (cons (f (car lst)) (map f (cdr lst))))))

;; Accepts a function and a list of numbers, and returns the elements of the list that satisfy the function.
;; filter : (Number -> Boolean) List -> List
(define filter (lambda (f lst)
    (if (null? lst)
        '()
        (if (f (car lst))
            (cons (car lst) (filter f (cdr lst)))
            (filter f (cdr lst))))))

;; Accepts two lists and returns a list of pairs of elements from the two lists.
;; zip : List List -> List
(define zip (lambda (lst1 lst2)
    (if (or (null? lst1) (null? lst2))
        '()
        (cons (list (car lst1) (car lst2)) (zip (cdr lst1) (cdr lst2))))))

;; Accepts three numbers: start, stop and step and returns a list of numbers from start to stop with the given step.
;; range : Number Number Number -> List
(define range (lambda (start stop step)
    (if (> start stop)
    '()
    (cons start (range (+ start step) stop step)))))

;; Accepts a list and returns true if the list is a palindrome and false otherwise.
;; is-palindrome: List -> Boolean
(define is-palindrome (lambda (lst)
    (equal? lst (reverse lst))))

;; Provide statements
(provide (all-defined-out))