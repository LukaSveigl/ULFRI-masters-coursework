#lang racket

(require "06.rkt")

(displayln "Test power function")
(displayln (equal? (power 2 3) 8))
(displayln (equal? (power 5 0) 1))
(displayln (equal? (power 3 2) 9))
(displayln (equal? (power 7 1) 7))

(displayln "Test gcd function")
(displayln (equal? (gcd 48 18) 6))
(displayln (equal? (gcd 101 10) 1))
(displayln (equal? (gcd 56 98) 14))
(displayln (equal? (gcd 20 8) 4))

(displayln "Test fib function")
(displayln (equal? (fib 1) 1))
(displayln (equal? (fib 2) 1))
(displayln (equal? (fib 3) 2))
(displayln (equal? (fib 4) 3))
(displayln (equal? (fib 5) 5))

(displayln "Test reverse function")
(displayln (equal? (reverse '(1 2 3 4)) '(4 3 2 1)))
(displayln (equal? (reverse '()) '()))
(displayln (equal? (reverse '(a b c)) '(c b a)))

(displayln "Test remove function")
(displayln (equal? (remove 2 '(1 2 3 2 4)) '(1 3 4)))
(displayln (equal? (remove 5 '(1 2 3 4)) '(1 2 3 4)))
(displayln (equal? (remove 1 '(1 1 1 1)) '()))

(displayln "Test map function")
(displayln (equal? (map (lambda (x) (* x x)) '(1 2 3 4)) '(1 4 9 16)))
(displayln (equal? (map (lambda (x) (+ x 1)) '(1 2 3 4)) '(2 3 4 5)))
(displayln (equal? (map (lambda (x) (* x 2)) '()) '()))

(displayln "Test filter function")
(displayln (equal? (filter (lambda (x) (> x 2)) '(1 2 3 4 5)) '(3 4 5)))
(displayln (equal? (filter (lambda (x) (even? x)) '(1 2 3 4 5)) '(2 4)))
(displayln (equal? (filter (lambda (x) (< x 0)) '(1 2 3 4 5)) '()))

(displayln "Test zip function")
(displayln (equal? (zip '(1 2 3) '(4 5 6)) '((1 4) (2 5) (3 6))))
(displayln (equal? (zip '(1 2) '(3 4 5)) '((1 3) (2 4))))
(displayln (equal? (zip '() '(1 2 3)) '()))

(displayln "Test range function")
(displayln (equal? (range 1 5 1) '(1 2 3 4 5)))
(displayln (equal? (range 0 10 2) '(0 2 4 6 8 10)))

(displayln "Test is-palindrome function")
(displayln (equal? (is-palindrome '(1 2 3 2 1)) #t))
(displayln (equal? (is-palindrome '(1 2 3 4 5)) #f))
(displayln (equal? (is-palindrome '()) #t))
(displayln (equal? (is-palindrome '(a b a)) #t))
(displayln (equal? (is-palindrome '(a b c)) #f))