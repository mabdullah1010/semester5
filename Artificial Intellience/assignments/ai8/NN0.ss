:: Muhammad Abdullah
;; AI COM 316
;; November 18th 2026
;; NN0.ss

;; uncomment the desired function

;; represents AND
(define threshold-weights '(((1.5 1 1))))



;; XOR
;(define threshold-weights '(((.6 1 -.5) (.6 -.5 1))
;                           ((.4 1 1))))

;; OR
;(define threshold-weights '(((0.5 1 1))))

;; 0 0 -> 0(1) + 0(1) - 0.5 = -0.5 = 0
;; 0 1 -> 0(1) + 1(1) - 0.5 = 0.5 = 1
;; 1 0 -> 1(1) + 0(1) -0.5 = 0.5 = 1
;; 1 1 -> 1(1) + 1(1) -0.5 = 1.5 = 1



;; NOT XOR
;(define threshold-weights '(((1.5 1 1) (-0.5 -1 -1))
;                           ((0.5 1 1))))

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