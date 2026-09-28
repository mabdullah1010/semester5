;;Muhammad Abdullah
;;September 19th 2026
;;AI2

(define h
  (lambda (point)
    (+ (abs (- (car point) (car goal)))
       (abs (- (cadr point) (cadr goal))))))



(define path-lst '())
(define frontier '()) ; Our priority queue

;; Manhattan Distance - used as heuristic throughout
(define h
  (lambda (point)
    (+ (abs (- (car point) (car goal)))
       (abs (- (cadr point) (cadr goal))))))

;; insertion cuz of h(n)
(define insert-best
  (lambda (nodes queue)
    (if (null? nodes)
        queue
        (insert-best (cdr nodes) (insert-one-best (car nodes) queue)))))

(define insert-one-best
  (lambda (node queue)
    (cond
      [(null? queue) (list node)]
      [(< (h node) (h (car queue))) (cons node queue)]
      [else (cons (car queue) (insert-one-best node (cdr queue)))])))

(define expand-best 
  (lambda (point)
    (let ((lst (adjacentv point)))
      (set-lst-visited lst)
      (add-to-path-lst lst point)
      ;; insert sorted
      (set! frontier (insert-best lst frontier)))))

(define search-best
  (lambda (grid stop-count)
    (block-set! start visited)
    (set! path-lst (list (list start '())))
    (set! frontier '())
    (search2-best grid 1 stop-count)))

(define search2-best
  (lambda (grid count stop-count)
    (pause pause-num)
    (display count) (newline)
    
    (expand-best robot)
    
    (let ((next-robot (if (null? frontier) '() (car frontier))))
      (cond
        [(null? next-robot) (display "no path found\n") #f]

        [(equal? next-robot goal) 
         (display "goal found\n")
         
         ;; move robot before ending
         (set! robot next-robot)
         (draw-moved-robot (car robot) (cadr robot))
         
         (draw-path (get-path goal)) #t]

        [(>= count stop-count) (display "stop count reached\n") #f]
        [else
         ;; remove from front of sorted frontier
         (set! frontier (cdr frontier))
         (set! robot next-robot)
         (draw-moved-robot (car robot) (cadr robot))
         (draw-visited (car robot) (cadr robot))
         (search2-best grid (+ count 1) stop-count)]))))


(define add-to-path-lst
  (lambda (lst point)
    (if (not (null? lst))
       (let ((child-parent (list (car lst) point)))
         (set! path-lst (cons child-parent path-lst))
         (add-to-path-lst (cdr lst) point)))))

(define set-lst-visited 
  (lambda (lst)
    (if (null? lst)
        '()
    ;else
        (let ((x (car lst)))
          (draw-pt-frontier x)
          (block-set! x visited)
          (set-lst-visited (cdr lst))))))
  
(define draw-pt-frontier
  (lambda (pt)
    (draw-frontier (car pt) (cadr pt))))