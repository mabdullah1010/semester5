;Muhammad Abdullah
;September 7th 2026
;scheme3

; grid-search.ss

(define none -1)
(define n 0)
(define s 1)
(define e 2)
(define w 3)

(define search
  (lambda (grid stop-count)
    (search2 grid 1 stop-count)))

(define search2
  (lambda (grid count stop-count)
    (pause pause-num)
    (let ((x (robot-x))
          (y (robot-y)))
      (display count)
      (newline)
      (mark-visited grid x y)
      (move-robot grid x y 0)
      (draw-moved-robot (robot-x) (robot-y))
      (if (or
            (and (= x (robot-x)) (= y (robot-y)))
            (equal? robot goal)
            (>= count stop-count))
         #f
       ;else
         (search2 grid (+ count 1) stop-count)))))

; mark visited square

(define mark-visited
  (lambda (grid x y)
    (cond
      ((= (get-node grid x y) free)
        (set-node! grid x y visited)
        (draw-visited x y)))))



(define open?  ; not an obstacle
  (lambda (grid x y)
    (and (>= x 0)
         (< x num-col-row)
         (>= y 0)
         (< y num-col-row)
         (< (get-node grid x y) obstacle))))


(define unvisited? ; open and not visited yet
  (lambda (grid x y)
    (and (open? grid x y)
         (< (get-node grid x y) visited))))

(define any-unvisited?
  (lambda (grid x y)
    (or (unvisited? grid (- x 1) y)
        (unvisited? grid (+ x 1) y)
        (unvisited? grid x (- y 1))
        (unvisited? grid x (+ y 1)))))


; pick neighbor not visited
; otherwise any square that is not an obstacle

(define move-robot ;
  (lambda (grid x y count)
    (if (any-unvisited? grid x y)
       (move-unvisited grid x y count)
       (move-open grid x y count))))

(define move-unvisited
  (lambda (grid x y count)
    (let ((dir (random 4)))
      (cond
        ((and (= dir n) (unvisited? grid (- x 1) y))
           (set! robot (list (- x 1) y)))
        ((and (= dir s) (unvisited? grid (+ x 1) y))
           (set! robot (list (+ x 1) y)))
        ((and (= dir w) (unvisited? grid x (- y 1)))
           (set! robot (list x (- y 1))))
        ((and (= dir e) (unvisited? grid x (+ y 1)))
           (set! robot (list x (+ y 1))))
        ((> count 100)
           (move-any-unvisited grid x y))
        (else
          (move-unvisited grid x y (+ count 1)))))))

(define move-open
  (lambda (grid x y count)
    (let ((dir (random 4)))
      (cond
        ((and (= dir n) (open? grid (- x 1) y))
           (set! robot (list (- x 1) y)))
        ((and (= dir s) (open? grid (+ x 1) y))
           (set! robot (list (+ x 1) y)))
        ((and (= dir w) (open? grid x (- y 1)))
           (set! robot (list x (- y 1))))
        ((and (= dir e) (open? grid x (+ y 1)))
           (set! robot (list x (+ y 1))))
        ((> count 100)
           (move-any-dir grid x y))
        (else
          (move-open grid x y (+ count 1)))))))

;fallback, used when the random draw keeps missing

(define move-any-unvisited
  (lambda (grid x y)
    (cond
      ((unvisited? grid (- x 1) y)
         (set! robot (list (- x 1) y)))
      ((unvisited? grid (+ x 1) y)
         (set! robot (list (+ x 1) y)))
      ((unvisited? grid x (- y 1))
         (set! robot (list x (- y 1))))
      ((unvisited? grid x (+ y 1))
         (set! robot (list x (+ y 1))))
      (else
        (move-any-dir grid x y)))))

(define move-any-dir
  (lambda (grid x y)
    (cond
      ((open? grid (- x 1) y)
         (set! robot (list (- x 1) y)))
      ((open? grid (+ x 1) y)
         (set! robot (list (+ x 1) y)))
      ((open? grid x (- y 1))
         (set! robot (list x (- y 1))))
      ((open? grid x (+ y 1))
         (set! robot (list x (+ y 1))))
      (else
        (display "no move")))))

(define pause
  (lambda (count)
    (if (<= count 0)
       0
     ;else
       (pause (- count 1)))))
