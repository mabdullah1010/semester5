;; Muhammad Abdullah; Abdul Rehman; Joseph Coombs
;; Robot
;; Monte Carlo Tree search
;; October 7th 2026

(define make-node 
  (lambda (state parent untried-moves)
    (vector state parent '() untried-moves 0 0.0)))

(define node-state      (lambda (n) (vector-ref n 0)))
(define node-parent     (lambda (n) (vector-ref n 1)))
(define node-children   (lambda (n) (vector-ref n 2)))
(define node-untried    (lambda (n) (vector-ref n 3)))
(define node-visits     (lambda (n) (vector-ref n 4)))
(define node-wins       (lambda (n) (vector-ref n 5)))

;; functions for the node vector
(define set-node-children! (lambda (n c) (vector-set! n 2 c)))
(define set-node-untried!  (lambda (n u) (vector-set! n 3 u)))
(define set-node-visits!   (lambda (n v) (vector-set! n 4 v)))
(define set-node-wins!     (lambda (n w) (vector-set! n 5 w)))

;; retrieve adjacent open blocks and staying in place
(define get-valid-moves
  (lambda (pos)
    (append (adjacento pos) (list pos))))



;; Calculates adversarial UCB1
;; If the Robot's turn, maximizes Robot wins
;; If Goal's turn, minimizes Robot wins.

(define ucb-score
  (lambda (child parent-visits is-robot-deciding)
    (let ((v (node-visits child))
          (w (node-wins child)))
      (if (= v 0)
          1000000.0 ;; Guarantee exploration of unvisited nodes
          (let ((win-rate (if is-robot-deciding
                              (/ w v)
                              (/ (- v w) v))))
            (+ win-rate (* 0.5 (sqrt (/ (log parent-visits) v)))))))))



;; pciks children with the highest UCB score
;; stops when it finds a node with untried moves or a terminal state

(define select-node
  (lambda (node)
    (cond

      ((equal? (car (node-state node)) (cadr (node-state node))) node)
      ((not (null? (node-untried node))) node)
      ((null? (node-children node)) node)
      (else (select-node (get-best-ucb-child node))))))




(define get-best-ucb-child
  (lambda (node)
    (let ((children (node-children node))
          (p-visits (node-visits node))

          (is-r-turn (caddr (node-state node)))) 
      (let loop ((c children) (best-child (car children)) (best-score -1.0))
        (if (null? c)
            best-child

            (let ((score (ucb-score (car c) p-visits is-r-turn)))
              (if (> score best-score)
                  (loop (cdr c) (car c) score)
                  (loop (cdr c) best-child best-score))))))))



(define expand-node
  (lambda (node)
    (if (or (null? (node-untried node))
            (equal? (car (node-state node)) (cadr (node-state node))))
        node 
        (let* ((move (car (node-untried node)))
               (rest-untried (cdr (node-untried node)))
               (state (node-state node))
               (r-pos (car state))
               (g-pos (cadr state))
               (is-r-turn (caddr state))
               
               ;; update board state
               (new-state (if is-r-turn
                              (list move g-pos #f)
                              (list r-pos move #t)))
                              
               (new-untried (if is-r-turn
                                (get-valid-moves g-pos)
                                (get-valid-moves r-pos)))
                                
               (new-node (make-node new-state node new-untried)))
               
          (set-node-untried! node rest-untried)
          (set-node-children! node (cons new-node (node-children node)))
          new-node))))


;; simluate the moves

(define simulate
  (lambda (node)
    (let loop ((state (node-state node)) (depth 0))
      (let ((r-pos (car state))
            (g-pos (cadr state))
            (is-r-turn (caddr state)))

        (cond
          ((equal? r-pos g-pos) 1.0)   ;; Win
          ((>= depth 300) 0.0)         ;; run out of moves (Loss)
          (is-r-turn
           (let* ((moves (get-valid-moves r-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list move g-pos #f) (+ depth 1))))

          (else
           (let* ((moves (get-valid-moves g-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list r-pos move #t) (+ depth 1)))))))))



(define backpropagate
  (lambda (node result)
    (if (not (null? node))

        (begin
          (set-node-visits! node (+ (node-visits node) 1))
          (set-node-wins! node (+ (node-wins node) result))
          (backpropagate (node-parent node) result)))))





(define get-next-robot
  (lambda (current-robot)
    (let* ((initial-untried (get-valid-moves current-robot))
           (root (make-node (list current-robot goal #t) '() initial-untried)))
      

      (let loop ((i 0))
        (if (< i 1000)
            (let* ((selected (select-node root))
                   (expanded (expand-node selected))
                   (result (simulate expanded)))
              (backpropagate expanded result)
              (loop (+ i 1)))))
      
      ;; pick the immediate physical child node to be visited more often
      (let ((children (node-children root)))
        (if (null? children)
            current-robot 

            (let loop2 ((c children) (best-child (car children)) (max-v -1))
              (if (null? c)

                  (car (node-state best-child))
                  (let ((v (node-visits (car c))))
                    (if (> v max-v)
                        (loop2 (cdr c) (car c) v)
                        (loop2 (cdr c) best-child max-v))))))))))