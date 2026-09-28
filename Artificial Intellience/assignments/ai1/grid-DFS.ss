(define path-lst '())


(define expand-dfs 
  (lambda (point)
    (let ((lst (adjacentv point)))
      (set-lst-visited lst)
      (add-to-path-lst lst point)
      (push lst)))) ;; push



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

(define search-dfs
  (lambda (grid stop-count)
    (block-set! start visited)
    (set! path-lst (list (list start '())))
    (search2-dfs grid 1 stop-count)))

(define search2-dfs
  (lambda (grid count stop-count)
    (pause pause-num)
    (display count)
    (newline)
    
    (expand-dfs robot)
    
    ;;look at the top of the stack
    (let ((next-robot (top))) 
      (cond
        [(null? next-robot) 
         (display "No path\n") #f]
         
        [(equal? next-robot goal) 
         (display "Goal found\n")
         (draw-path (get-path goal)) #t]
         
        [(>= count stop-count) 
         (display "Stop count\n") #f]
         
        [else
         ;; remove from stack
         (pop) 
         (set! robot next-robot)
         (draw-moved-robot (car robot) (cadr robot))
         (draw-visited (car robot) (cadr robot))
         
         (search2-dfs grid (+ count 1) stop-count)]))))


    
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