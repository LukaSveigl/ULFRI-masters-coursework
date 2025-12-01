#lang racket

(require "07.rkt")

(displayln "Test ones stream")
(displayln (equal? (car ((cdr ((cdr ones))))) 1))
(displayln (equal? (car ones) 1))

(displayln "Test naturals stream")
(displayln (equal? (car naturals) 1))
(displayln (equal? (car ((cdr ((cdr ones))))) 3))

(displayln "Test fibs stream")
(displayln (equal? (car ((cdr ((cdr fibs))))) 2))
(displayln (equal? (car ((cdr fibs))) 1))

(displayln "Test first function")
(displayln (equal? (first 5 fibs) '(1 1 2 3 5)))
(displayln (equal? (first 5 naturals) '(1 2 3 4 5)))

(displayln "Test squares function")
(displayln (equal? (car ((cdr ((cdr (squares ones)))))) 1))
(displayln (equal? (car (squares ones)) 1))
(displayln (equal? (car ((cdr ((cdr (squares naturals)))))) 9))

(displayln "Test sml macro")
(displayln (equal? (sml nil) '())) ; Test nil
(displayln (equal? (sml null (sml nil)) #t)) ; Test null
(displayln (equal? (sml null (sml :: 1 (sml nil))) #f)) ; Test null with non-empty list
(displayln (equal? (sml hd (sml :: 5 (sml nil))) 5)) ; Test hd
(displayln (equal? (sml tl (sml :: 5 (sml :: 4 (sml nil)))) (sml :: 4 (sml nil)))) ; Test tl
(displayln (equal? (sml tl (sml :: 5 (sml nil))) (sml nil))) ; Test tl with single element list
(displayln (equal? (sml :: 1 (sml :: 2 (sml :: 3 (sml nil)))) (cons 1 (cons 2 (cons 3 '()))))) ; Test ::
(displayln (equal? (sml hd (sml :: 1 (sml :: 2 (sml :: 3 (sml nil))))) 1)) ; Test hd with multiple elements
(displayln (equal? (sml tl (sml :: 1 (sml :: 2 (sml :: 3 (sml nil))))) (sml :: 2 (sml :: 3 (sml nil))))) ; Test tl with multiple elements
(displayln (equal? (sml hd (sml tl (sml :: 1 (sml :: 2 (sml :: 3 (sml nil)))))) 2)) ; Test hd after tl
(displayln (equal? (sml tl (sml tl (sml :: 1 (sml :: 2 (sml :: 3 (sml nil)))))) (sml :: 3 (sml nil)))) ; Test tl after tl
(displayln (equal? (sml null (sml tl (sml tl (sml tl (sml :: 1 (sml :: 2 (sml :: 3 (sml nil)))))))) #t)) ; Test null after multiple tls
(displayln (sml 5 :: null))

(displayln "Test my-delay and my-force")
(define delayed (my-delay (+ 1 2)))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3)) ; Should recompute here
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3))
(displayln (equal? (my-force delayed) 3)) ; Should recompute here

(displayln "Test partitions function")
(displayln (equal? (partitions 3 5) 2))
(displayln (equal? (partitions 4 10) 5))
(displayln (equal? (partitions 0 0) 1))
(displayln (equal? (partitions -1 5) 0))
(displayln (equal? (partitions 3 -5) 0))