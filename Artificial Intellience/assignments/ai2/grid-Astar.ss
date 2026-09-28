;;Muhammad Abdullah
;;September 19th 2026
;;AI2


(define path-lst '())
(define frontier '()) 

(define h
  (lambda (point)
    (+ (abs (- (car point) (car goal)))
       (abs (- (cadr point) (cadr goal))))))

;; g(n).... cost so far (length of the path back to start)
(define g
  (lambda (point)
    (length (get-path point))))

;; f(n) = g(n) + h(n)
(define f
  (lambda (point)
    (+ (g point) (h point))))

;; insertion cuz of f(n)
(define insert-astar
  (lambda (nodes queue)
    (if (null? nodes)
        queue
        (insert-astar (cdr nodes) (insert-one-astar (car nodes) queue)))))

(define insert-one-astar
  (lambda (node queue)
    (cond
      [(null? queue) (list node)]
      [(< (f node) (f (car queue))) (cons node queue)]
      [else (cons (car queue) (insert-one-astar node (cdr queue)))])))

;; expand
(define expand-astar 
  (lambda (point)
    (let ((lst (adjacentv point)))
      (set-lst-visited lst)
      (add-to-path-lst lst point)
      (set! frontier (insert-astar lst frontier)))))

(define search-astar
  (lambda (grid stop-count)
    (block-set! start visited)
    (set! path-lst (list (list start '())))
    (set! frontier '())
    (search2-astar grid 1 stop-count)))

(define search2-astar
  (lambda (grid count stop-count)
    (pause pause-num)
    (display count) (newline)
    
    (expand-astar robot)
    
    (let ((next-robot (if (null? frontier) '() (car frontier))))
      (cond
        [(null? next-robot) (display "no path found\n") #f]

        [(equal? next-robot goal) 
         (display "goal found\n")
         
         ;; move to goal before enfing
         (set! robot next-robot)
         (draw-moved-robot (car robot) (cadr robot))
         
         (draw-path (get-path goal)) #t]

        [(>= count stop-count) (display "stop count reached\n") #f]
        [else
         (set! frontier (cdr frontier))
         (set! robot next-robot)
         (draw-moved-robot (car robot) (cadr robot))
         (draw-visited (car robot) (cadr robot))
         (search2-astar grid (+ count 1) stop-count)]))))


(define find-parent
  (lambda (node lst)
    (cond
      [(null? lst) #f] 
      
      [(equal? node (caar lst)) (cadar lst)] 
      
      [else (find-parent node (cdr lst))])))


(define get-path
  (lambda (current-node)
    (if (equal? current-node start)
        (list start)
        (let ((parent (find-parent current-node path-lst)))
          (append (get-path parent) (list current-node))))))

      
(define draw-path
  (lambda (path)
    (cond 
      ((not (null? path))
         (draw-pt-path-node (car path))
         (draw-path (cdr path))))))
 
(define draw-pt-path-node
  (lambda (point)
    (draw-path-node (car point) (cadr point))))


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