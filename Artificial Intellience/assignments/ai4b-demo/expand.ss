(define r-node-position
  (lambda (node)
    (let ((state (node-state node)))
      (if (caddr state) (car state) (cadr state)))))

(define r-build-child-state
  (lambda (node move)
    (let* ((state (node-state node))
           (r-pos (car state))
           (g-pos (cadr state))
           (is-r-turn (caddr state)))
      (if is-r-turn
          (list move g-pos #f)
          (list r-pos move #t)))))

(define r-expand
  (lambda (node)
    (cond
      ((equal? (car (node-state node)) (cadr (node-state node))) node)
      (else
        (begin
          (if (and (null? (node-untried node)) (null? (node-children node)))
              (set-node-untried! node (r-get-valid-moves (r-node-position node))))
          (let* ((move (car (node-untried node)))
                 (child-state (r-build-child-state node move))
                 (child (vector child-state node '() '() 0 0)))
            (set-node-untried! node (cdr (node-untried node)))
            (set-node-children! node (cons child (node-children node)))
            child))))))

; Calculates the UCB score for a node
;num-wins = number of simulations won from this node
;num-visits = number of times this node has been visited
;parent-visits = number of times the parent node has been visited

(define r-infinity 1000000000000)

(define (r-ucb-score num-wins num-visits parent-visits)
  (if (= num-visits 0)
      r-infinity
      ; else
      (+ (/ num-wins num-visits)
       (* 2
        (sqrt (/ (log parent-visits)
               num-visits))))))