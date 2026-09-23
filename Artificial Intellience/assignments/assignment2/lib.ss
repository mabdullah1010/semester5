;Muhammad Abdullah
;September 7th 2026

;scheme 2


(define (sqr-list lst)
  (if (null? lst) '()
      (cons (* (car lst) (car lst)) ;multiply and add to list
        (sqr-list (cdr lst))))) ; move to next element



(define (place lst x)
  (cond
    [(null? lst) (list x)] ; return x in list is list is null
    [(<= x (car lst)) (cons x lst)] ; add at correct position
    [else (cons (car lst) (place (cdr lst) x))])) ; keep chekcing from next element

(define (change x)
  (cond
    [(< x 0) (* x x)]
    [(= 0 x) x]
    [(> x 0) (+ x 1)]))


(define (change-list lst)
  (map(lambda (x) (change x)) lst))



(define (dist-sq p1 p2)
  (let ([dx (- (car p1) (car p2))]
        [dy (- (cadr p1) (cadr p2))])
    (+ (* dx dx) (* dy dy)))) ; euqlidean distance without square root to simplify


(define (closest-point pt lst)
  (if (null? lst)
      #f
      (let loop ([rest (cdr lst)]
                 [closest (car lst)]
                 [min-dist (dist-sq pt (car lst))]) ; initialize
        (if (null? rest)
            closest
            (let* ([current (car rest)]
                   [current-dist (dist-sq pt current)])
              (if (< current-dist min-dist)
                  (loop (cdr rest) current current-dist) ;update
                  (loop (cdr rest) closest min-dist)))))))



(define (add-list lst1 lst2)
  (cond
    [(null? lst1) lst2]
    [(null? lst2) lst1]
    [else (cons (+ (car lst1) (car lst2)) 
                (add-list (cdr lst1) (cdr lst2)))]))
                
                

(define (delete-lists lst)
  (cond
    [(null? lst) '()]
    [(not (list? (car lst))) 
     (cons (car lst) (delete-lists (cdr lst)))] ; add element
    [else
      (delete-lists (cdr lst))])) ; skip element

(define (flatten lst)
  (cond
    [(null? lst) '()]
    [(list? (car lst)) ; check next element
     (append (flatten (car lst)) (flatten (cdr lst)))]
    [else
      (cons (car lst) (flatten (cdr lst)))]))




(vector 1 2 3 4 5) ; creates a new vector

(vector-length '#(1 2 3 4 5)) ; returns total number of elements in the vector

(vector-ref '#(1 2 3 4 5) 2) ; retrieves element at index 2

(vector->list '#(a b c d e f)) ; converts vector into a list

(list->vector '(a b c d e f)) ; Converts list into a vectorr

(vector-map abs '#(1 -2 3 -4 5 -6)) ; applies 'abs' to each element to absolute values

(vector-map (lambda (x y) (* x y)) '#(1 2 3 4) '#(7 8 9 10)) ; multiplies parallel elements

(let ([v (vector 1 2 3)]) (vector-set! v 1 'two) v) ; changes index 1 to 'two

(vector-for-each (lambda (x y) (when (< x y) (display x))) '#(1 2 3) '#(4 1 3)) ; prints elements from first vector that are smaller than second

(vector? '#(1 2 3)) ; checks if the object is a vector

(vector-length '#(a b c)) ; returns the length of the vector

(vector-ref '#(10 20 30) 0) ; retrieves the first element (index 0)

(let ([v (vector 'a 'b 'c)]) (vector->list v)) ; converts a local vector into a list

(let ([lst '(1 2 3)]) (list->vector lst)) ; Converts local list into a vector

(vector-map - '#(1 2 3)) ; makes each value negative

(let ([v (vector 10 20)]) (vector-set! v 0 99) v) ; Updates index 0 to 99

(vector? '(1 2 3)) ; checks if a list is a vector and returns #f

(vector-map + '#(1 2) '#(3 4)) ; adds corresponding elements

(vector->list (vector 1 2)) ; converts a new vector into a list 

(list->vector (vector->list '#(1 2 3))) ; converts a vector to a list and again to a vector 
