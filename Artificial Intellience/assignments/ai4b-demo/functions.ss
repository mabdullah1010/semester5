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

(define r-get-valid-moves
  (lambda (pos)
    (append (adjacento pos) (list pos))))


(define r-ucb-score
  (lambda (child parent-visits is-robot-deciding)
    (let ((v (node-visits child))
          (w (node-wins child)))
      (if (= v 0)
          1000000.0 ;; exploration of unvisited nodes
          (let ((win-rate (if is-robot-deciding
                              (/ w v)
                              (/ (- v w) v))))
            (+ win-rate (* 0.5 (sqrt (/ (log parent-visits) v)))))))))


(define r-select-node
  (lambda (node)
    (cond
      ((equal? (car (node-state node)) (cadr (node-state node))) node)
      ((not (null? (node-untried node))) node)
      ((null? (node-children node)) node)
      (else (r-select-node (r-get-best-ucb-child node))))))

(define r-get-best-ucb-child
  (lambda (node)
    (let ((children (node-children node))

          (p-visits (node-visits node))

          ;; sees whose turn it is in the current node (to pass to UCB)
          (is-r-turn (caddr (node-state node)))) 

      (let loop ((c children) (best-child (car children)) (best-score -1.0))
        (if (null? c)

            best-child
            (let ((score (r-ucb-score (car c) p-visits is-r-turn)))
              (if (> score best-score)
                  (loop (cdr c) (car c) score)

                  (loop (cdr c) best-child best-score))))))))



(define r-simulate
  (lambda (node)
    (let loop ((state (node-state node)) (depth 0))
      (let ((r-pos (car state))
            (g-pos (cadr state))
            (is-r-turn (caddr state)))
        (cond
          ((equal? r-pos g-pos) 1.0)   ;; Win
          ((>= depth 300) 0.0)         ;; Loss

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

          (set-node-visits! node (+ (node-visits node) 1))
          (set-node-wins! node (+ (node-wins node) result))
          
          (r-backpropagate (node-parent node) result)))))
