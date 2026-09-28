;; COM 316 - AI
;; Muhammad Abdullah
;; September 26, 2026
;; Realtime A Star


(define h-matrix '())

(define init-h-matrix
  (lambda ()
    (set! h-matrix (make-vector num-col-row))
    (init-h-matrix-helper 0)))
    
(define init-h-matrix-helper
  (lambda (y)
    (if (< y num-col-row)
        (begin
          (vector-set! h-matrix y (make-vector num-col-row -1))
          (init-h-matrix-helper (+ y 1))))))

;; get heuristic (updated memory if it exists, else manhattan distance)

(define get-h
  (lambda (pt)
    (let ((val (vector-ref (vector-ref h-matrix (cadr pt)) (car pt))))
      (if (= val -1)
          (+ (abs (- (car pt) (car goal)))
             (abs (- (cadr pt) (cadr goal))))
          val))))

;; set heuristic by overwriting 
(define set-h!
  (lambda (pt val)
    (vector-set! (vector-ref h-matrix (cadr pt)) (car pt) val)))

;; filters out walls
;; allows visited blocks so the robot can backtrack

(define filter-obs
  (lambda (lst)
    (cond
      [(null? lst) '()]
      [(not (= (block-status (car lst)) obstacle)) 
       (cons (car lst) (filter-obs (cdr lst)))]
      [else (filter-obs (cdr lst))])))

;; f(n) = 1 + h(n) 
(define evaluate-neighbors
  (lambda (neighbors)
    (if (null? neighbors)
        '()
        (let* ((n (car neighbors))
               (f-val (+ 1 (get-h n))))
          (cons (list n f-val) (evaluate-neighbors (cdr neighbors)))))))




          
;; order neighbors from lowest f(n) to highest
(define insert-evaled

  (lambda (pair lst)

    (cond
      [(null? lst) (list pair)]
      [(< (cadr pair) (cadr (car lst))) (cons pair lst)] 
      [else (cons (car lst) (insert-evaled pair (cdr lst)))])))
      



(define sort-evaled
  (lambda (lst)
    (if (null? lst)
        '()

        (insert-evaled (car lst) (sort-evaled (cdr lst))))))



(define search-rta
  (lambda (grid stop-count)

    (init-h-matrix)

    (search2-rta grid 0 stop-count)))
    
(define search2-rta
  (lambda (grid steps stop-count)
    (pause pause-num)
    
    (cond
      [(equal? robot goal) 
       (display "goal found in ") (display steps) (display " steps\n") #t]
       
      [(>= steps stop-count) 
       (display "stop count reached\n") #f]
       
      [else
       (let* ((neighbors (filter-obs (adjacent robot)))
              (evaled (evaluate-neighbors neighbors))
              (sorted (sort-evaled evaled)))
         
         (if (null? sorted)
             (begin (display "Trapped :( :( \n") #f)
             (let* ((best-pair (car sorted))
                    (best-node (car best-pair))
                    ;; if a dead end, penalize heavily
                    ;; otherwise, set to the second-best option

                    
                    (second-best-f (if (> (length sorted) 1)
                                       (cadr (cadr sorted))
                                       obstacle)))
               
               ;; overwrite current block's heuristic with the penalty
               (set-h! robot second-best-f)
               
               ;; mark visited 
               (draw-visited (car robot) (cadr robot))
               
               ;; move robot
               (set! robot best-node)
               (draw-moved-robot (car robot) (cadr robot))
               
               (search2-rta grid (+ steps 1) stop-count))))])))