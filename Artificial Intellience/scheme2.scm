(define (sqr-list lst)
  (if (null? lst) '()
      (cons (* (car lst) (car lst))
        (sqr-list (cdr lst)))))



(define (place lst x)
  (cond
    [(null? lst) (list x)]
    [(<= x (car lst)) (cons x lst)]
    [else (cons (car lst) (place (cdr lst) x))]))

(define (change x)
  (cond
    [(< x 0) (* x x)]
    [(= 0 x) x]
    [(> x 0) (+ x 1)]))


(define (change-list lst)
  (map(lambda (x) (change x)) lst))


(define (closest-point pt lst)
  (if (null? lst) #f
      (let loop ([remain (cdr lst)]
                 [closest (car lst)]
                 [min-dist (abs (- pt (car lst)))])
        (if (null? remain)
            closest
            (let* ([current (car remain)]
                   [current-dist (abs (- pt current))])
              (if (< current-dist min-dist)
                  (loop (cdr remain) current current-dist)
                  (loop (cdr remain) closest min-dist)))))))


(define (add-list lst1 lst2)
  (cond
    [(not (= (length lst1) (length lst2))) #f]
    [(null? lst1) '()]
    [else (cons (+ (car lst1) (car lst2))
            (add-list (cdr lst1) (cdr lst2)))]))


(define (delete-lists lst)
  (cond
    [(null? lst) '()]
    [(not (list? (car lst)))
     (cons (car lst) (delete-lists (cdr lst)))]
    [else
      (delete-lists (cdr lst))]))

(define (flatten lst)
  (cond
    [(null? lst) '()]
    [(list? (car lst))
     (append (flatten (car lst)) (flatten (cdr lst)))]
    [else
      (cons (car lst) (flatten (cdr lst)))]))


(vector 1 2 3 4 5)

(vector-length #(1 2 3 4 5))

(vector-ref #(1 2 3 4 5) 2)

(vector->list #(a b c d e f)) 

(list->vector '(a b c d e f))

(vector-map abs #(1 -2 3 -4 5 -6))

(vector-map (lambda (x y) (* x y)) '#(1 2 3 4) '#(7 8 9 10))

(let ([v (vector 1 2 3)]) (vector-set! v 1 'two) v)

(vector-for-each (lambda (x y) (when (< x y) (display x))) '#(1 2 3) '#(4 1 3))

(vector? #(1 2 3))

(vector-length #(a b c))

(vector-ref #(10 20 30) 0)

(let ([v (vector 'a 'b 'c)]) (vector->list v))

(let ([lst '(1 2 3)]) (list->vector lst))

(vector-map - #(1 2 3))

(let ([v (vector 10 20)]) (vector-set! v 0 99) v)

(vector? '(1 2 3))

(vector-map + #(1 2) #(3 4))

(vector->list (vector 1 2))

(list->vector (vector->list #(1 2 3)))

        