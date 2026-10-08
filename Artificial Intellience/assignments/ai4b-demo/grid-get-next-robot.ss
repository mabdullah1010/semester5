;; Muhammad Abdullah; Abdul Rehman; Joseph Coombs
;; Robot
;; Monte Carlo Tree search
;; October 7th 2026

(define r-make-node 
  (lambda (state parent untried-moves)
    (vector state parent '() untried-moves 0 0.0)))

(define r-node-state      (lambda (n) (vector-ref n 0)))
(define r-node-parent     (lambda (n) (vector-ref n 1)))
(define r-node-children   (lambda (n) (vector-ref n 2)))
(define r-node-untried    (lambda (n) (vector-ref n 3)))
(define r-node-visits     (lambda (n) (vector-ref n 4)))
(define r-node-wins       (lambda (n) (vector-ref n 5)))

(define r-set-node-children! (lambda (n c) (vector-set! n 2 c)))
(define r-set-node-untried!  (lambda (n u) (vector-set! n 3 u)))
(define r-set-node-visits!   (lambda (n v) (vector-set! n 4 v)))
(define r-set-node-wins!     (lambda (n w) (vector-set! n 5 w)))

;; helper functions

(define r-get-valid-moves
  (lambda (pos)
    (append (adjacento pos) (list pos))))

;; gets the player's position from the node state
(define r-node-position
  (lambda (node)
    (let ((state (r-node-state node)))
      (if (caddr state) (car state) (cadr state)))))


;; makes new state list after a move
(define r-build-child-state
  (lambda (node move)
  
    (let* ((state (r-node-state node))
           (r-pos (car state))
           (g-pos (cadr state))
           (is-r-turn (caddr state)))
      (if is-r-turn
          (list move g-pos #f)
          (list r-pos move #t)))))


(define r-infinity 1000000000000)

;; UCB1 score
(define r-ucb-score
  (lambda (num-wins num-visits parent-visits is-robot-deciding)
    (if (= num-visits 0)
        r-infinity
        (let ((win-rate (if is-robot-deciding
                            (/ num-wins num-visits)
                            (/ (- num-visits num-wins) num-visits))))
          (+ win-rate (* 1.414 (sqrt (/ (log parent-visits) num-visits))))))))


(define r-select-node
  (lambda (node)
    (cond
      ((equal? (car (r-node-state node)) (cadr (r-node-state node))) node)
      ;; if untried and children are both empty then stop

      ((and (null? (r-node-untried node)) (null? (r-node-children node))) node)
      ((not (null? (r-node-untried node))) node)
      (else (r-select-node (r-get-best-ucb-child node))))))


(define r-get-best-ucb-child
  (lambda (node)
    (let ((children (r-node-children node))
          (p-visits (r-node-visits node))
          (is-r-turn (caddr (r-node-state node)))) 
      (let loop ((c children) (best-child (car children)) (best-score -1.0))


        (if (null? c)
            best-child
            (let ((score (r-ucb-score (r-node-wins (car c)) 
                                      (r-node-visits (car c)) 
                                      p-visits 
                                      is-r-turn)))
              (if (> score best-score)
                  (loop (cdr c) (car c) score)
                  (loop (cdr c) best-child best-score))))))))



;; Expansion
(define r-expand
  (lambda (node)
    (cond
      ((equal? (car (r-node-state node)) (cadr (r-node-state node))) node)
      (else
        (begin
          (if (and (null? (r-node-untried node)) (null? (r-node-children node)))
              (r-set-node-untried! node (r-get-valid-moves (r-node-position node))))
          
          ;; prevent car on empty list if trapped
          (if (null? (r-node-untried node))
              node

              (let* ((move (car (r-node-untried node)))
                     (child-state (r-build-child-state node move))
                     (child (r-make-node child-state node '())))
                (r-set-node-untried! node (cdr (r-node-untried node)))
                (r-set-node-children! node (cons child (r-node-children node)))
                child)))))))



;; Simulation
(define r-simulate
  (lambda (node)
    (let loop ((state (r-node-state node)) (depth 0))
      (let ((r-pos (car state))
            (g-pos (cadr state))
            (is-r-turn (caddr state)))

        (cond
          ((equal? r-pos g-pos) 1.0)   ;; Win
          ((>= depth 80) 0.0)          ;; Loss (out of depth)
          (is-r-turn
           (let* ((moves (r-get-valid-moves r-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list move g-pos #f) (+ depth 1))))
          (else
           (let* ((moves (r-get-valid-moves g-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list r-pos move #t) (+ depth 1)))))))))


(define r-backpropagate
  (lambda (node result)
    (if (not (null? node))

        (begin
          (r-set-node-visits! node (+ (r-node-visits node) 1))
          (r-set-node-wins! node (+ (r-node-wins node) result))
          (r-backpropagate (r-node-parent node) result)))))



(define get-next-robot
  (lambda (current-robot)
    (let* ((initial-untried (r-get-valid-moves current-robot))
           (root (r-make-node (list current-robot goal #t) '() initial-untried)))
      

      (let loop ((start-time (real-time)))
        (let* ((selected (r-select-node root))
               (expanded (r-expand selected))
               (result (r-simulate expanded)))
          (r-backpropagate expanded result)
          (if (< (- (real-time) start-time) 4500)
              (loop start-time))))
      
      (let ((children (r-node-children root)))
        (if (null? children)
            current-robot 

            (let loop2 ((c children) (best-child (car children)) (max-v -1))
              (if (null? c)
                  (car (r-node-state best-child))
                  (let ((v (r-node-visits (car c))))
                    (if (> v max-v)
                        (loop2 (cdr c) (car c) v)
                        (loop2 (cdr c) best-child max-v))))))))))