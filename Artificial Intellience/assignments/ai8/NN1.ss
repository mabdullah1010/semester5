:: Muhammad Abdullah
;; AI COM 316
;; November 18th 2026
;; NN0.ss


;;  true around 0.993, false around 0.007
(define threshold-weights '(((15 10 10))))

;; OR
;(define threshold-weights '(((5 10 10))))

;; XOR
;(define threshold-weights '(((5 10 -10) (5 -10 10))
;                           ((5 10 10))))


(define NN
  (lambda (lst)
    (NN2 lst threshold-weights)))

(define NN2
  (lambda (lst tw)
    (display lst)
    (newline)
    (if (null? tw)
      lst
    ;else
      (let ((next-level (get-next-level lst (car tw))))
         (NN2 next-level (cdr tw))))))

(define get-next-level
  (lambda (lst twl)
     (if (null? twl)
       '()
     ;else
       (cons (get-node lst (car twl)) (get-next-level lst (cdr twl))))))

(define get-node
  (lambda (lst twn)
    (let ((threshold (car twn))
          (weights (cdr twn)))
      (g (+ (get-activations lst weights) (- threshold))))))

(define get-activations
  (lambda (lst w)
    (if (null? lst)
       0
    ;else
       (+ (* (car lst) (car w)) (get-activations (cdr lst) (cdr w))))))

(define g
  (lambda (x)
    (/ 1 (+ 1 (exp (- x))))))