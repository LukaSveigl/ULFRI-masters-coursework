#lang racket

;; Defines a stream of all ones.
(define ones (cons 1 (lambda () ones)))

;; Defines a stream-map function for streams.
(define (stream-map f stream)
  (cons (f (car stream))
        (lambda () (stream-map f ((cdr stream))))))

;; Defines a stream of all natural numbers.
(define naturals
  (cons 1
        (lambda ()
          (stream-map (lambda (x) (+ x 1)) naturals))))

;; tok fibs, ki ustreza zaporedju Fibonaccijevih števil (1 1 2 3 5 ...)
;; 
;; > (car ((cdr ((cdr fibs)))))
;; 2
(define fibs
  (letrec ((fibs-helper
            (lambda (a b)
              (cons a (lambda () (fibs-helper b (+ a b)))))))
    (fibs-helper 1 1)))


;; Defines a function first that returns the first n elements from the given stream.
(define (first n stream)
  (if (= n 0)
      '()
      (cons (car stream) (first (- n 1) ((cdr stream))))))

;; Defines a function that accepts a stream and returns the a new stream with squared elements.
(define (squares stream)
  (stream-map (lambda (x) (* x x)) stream))

;; makro sml, ki podpira uporabo "SML sintakse" za delo s seznami. Podprite SML funkcije/konstruktorje ::, hd, tl, null in nil. Sintaksa naj bo taka, kot je navedena v primeru uporabe spodaj. (Sintaksa seveda ne bo povsem enaka SML-jevi, saj zaradi zahtev Racketa še vedno ne smemo pisati odvečnih oklepajev, potrebno pa je pisati presledke okoli funkcij/parametrov, pa vseeno.)
;; 
;; > (sml nil)
;; '()
;; > (sml null (sml nil))
;; #t
;; > (sml hd (sml 5 :: null))
;; 5
;; > (sml tl (sml 5 :: (sml 4 :: (sml nil))))
;; '(4)
(define-syntax sml
  (syntax-rules (:: nil null hd tl)
    ((_ nil) '())
    ((_ null x) (null? x))
    ((_ hd x) (car x))
    ((_ tl x) (cdr x))
    ((_ x :: y) (cons x y))))
    
;; my-delay, my-force. Funkciji za zakasnitev in sprožitev delujeta tako, da si funkcija za sprožitev pri prvem klicu zapomni rezultat, ob naslednjih pa vrne shranjeno vrednost. Popravite funkciji tako, da bo funkcija za sprožitev ob prvem in nato ob vsakem petem klicu ponovno izračunala in shranila rezultat.
(define-syntax my-delay
  (syntax-rules ()
    ((_ x)
     (lambda ()
       (let ([f x])
         (let ([counter (make-parameter 0)])
           (lambda ()
             (if (= (counter) 0)
                 (begin
                   (counter 1)
                   (f))
                 (if (= (remainder (counter) 5) 0)
                     (begin
                       (counter 1)
                       (f))
                     (begin
                       (counter (+ (counter) 1))
                       (f)))))))))))

(define-syntax my-force
    (syntax-rules ()
        ((_ x) (x))))

;; partitions, ki sprejme števili k in n, ter vrne število različnih načinov, na katere lahko n zapišemo kot vsoto k naravnih števil (naravna števila se v tem kontekstu začnejo z 1). (Če se dva zapisa razlikujeta samo v vrstnem redu elementov vsote, ju obravnavamo kot en sam zapis. https://en.wikipedia.org/wiki/Partition_(number_theory))
(define (partitions k n)
  (cond
    ((or (< k 0) (< n 0)) 0) ; If k or n is negative, return 0
    ((or (= k 0) (= n 0)) 1) ; If k or n is 0, return 1
    ((= k 1) 1) ; If k is 1, return 1
    (else (+ (partitions (- k 1) n)
             (partitions k (- n k)))))) ; Recursive cases

;; Provide statements
(provide ones naturals fibs first squares sml my-delay my-force partitions)