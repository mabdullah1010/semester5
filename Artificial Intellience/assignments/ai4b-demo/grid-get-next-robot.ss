;; =======================================================
;; AI Problem: Game Playing (Pure Monte Carlo Tree Search)
;; Environment: 20x20 or 30x30 Grid
;; Agent: Robot (Chasing Goal)
;; 
;; This implementation utilizes a purely statistical MCTS approach 
;; without distance heuristics. It relies on a high-depth simulation 
;; (400 steps) on a smaller grid to organically map out intersections, 
;; paired with an adversarial UCB1 score to accurately model the goal's
;; evasion incentives.
;; =======================================================

;; -------------------------------------------------------
;; 1. NODE DATA STRUCTURE
;; Represents a distinct state in the Monte Carlo Tree.
;; Vector structure: #(state parent children untried-moves visits wins)
;; state = (list robot-pos goal-pos is-robot-turn)
;; -------------------------------------------------------
(define make-node 
  (lambda (state parent untried-moves)
    (vector state parent '() untried-moves 0 0.0)))

;; Accessor functions for the node vector
(define node-state      (lambda (n) (vector-ref n 0)))
(define node-parent     (lambda (n) (vector-ref n 1)))
(define node-children   (lambda (n) (vector-ref n 2)))
(define node-untried    (lambda (n) (vector-ref n 3)))
(define node-visits     (lambda (n) (vector-ref n 4)))
(define node-wins       (lambda (n) (vector-ref n 5)))

;; Mutator functions for the node vector
(define set-node-children! (lambda (n c) (vector-set! n 2 c)))
(define set-node-untried!  (lambda (n u) (vector-set! n 3 u)))
(define set-node-visits!   (lambda (n v) (vector-set! n 4 v)))
(define set-node-wins!     (lambda (n w) (vector-set! n 5 w)))

;; Helper to retrieve adjacent open blocks + staying in place
(define get-valid-moves
  (lambda (pos)
    (append (adjacento pos) (list pos))))

;; -------------------------------------------------------
;; 2. MCTS ENGINE FUNCTIONS
;; -------------------------------------------------------

;; Calculates adversarial UCB1. 
;; If it is the Robot's turn, it maximizes Robot wins. 
;; If it is the Goal's turn, it actively minimizes Robot wins, 
;; preventing the tree from expecting "suicidal" opponent mistakes.
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

;; PHASE 1: SELECTION (Tree Traversal)
;; Traverses down the tree by picking children with the highest UCB score.
;; Stops when it finds a node with untried moves or a terminal state.
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
          ;; Extracts whose turn it is in the current node to pass to UCB
          (is-r-turn (caddr (node-state node)))) 
      (let loop ((c children) (best-child (car children)) (best-score -1.0))
        (if (null? c)
            best-child
            (let ((score (ucb-score (car c) p-visits is-r-turn)))
              (if (> score best-score)
                  (loop (cdr c) (car c) score)
                  (loop (cdr c) best-child best-score))))))))

;; PHASE 2: EXPANSION
;; Pops an untried move from the selected node and creates a new child node,
;; maintaining strict physical geometry for the next player's valid moves.
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
               
               ;; Update board state based on whose turn just resolved
               (new-state (if is-r-turn
                              (list move g-pos #f)
                              (list r-pos move #t)))
                              
               ;; Generate valid untried moves strictly from the current location 
               ;; of the player whose turn is NEXT.
               (new-untried (if is-r-turn
                                (get-valid-moves g-pos)
                                (get-valid-moves r-pos)))
                                
               (new-node (make-node new-state node new-untried)))
               
          (set-node-untried! node rest-untried)
          (set-node-children! node (cons new-node (node-children node)))
          new-node))))

;; PHASE 3: SIMULATION (Pure Random Rollout)
;; Plays a fast-forward random game up to 400 steps. 
;; High depth ensures the random walks have sufficient stamina to intersect 
;; on a 20x20 or 30x30 board without returning zero prematurely.
(define simulate
  (lambda (node)
    (let loop ((state (node-state node)) (depth 0))
      (let ((r-pos (car state))
            (g-pos (cadr state))
            (is-r-turn (caddr state)))
        (cond
          ((equal? r-pos g-pos) 1.0)   ;; Terminal Win
          ((>= depth 300) 0.0)         ;; Stamina Limit (Loss)
          (is-r-turn
           (let* ((moves (get-valid-moves r-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list move g-pos #f) (+ depth 1))))
          (else
           (let* ((moves (get-valid-moves g-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list r-pos move #t) (+ depth 1)))))))))

;; PHASE 4: BACKPROPAGATION
;; Bubbles the 1.0 or 0.0 result back up the tree branch, incrementing 
;; visits and wins for every parent node back to the root.
(define backpropagate
  (lambda (node result)
    (if (not (null? node))
        (begin
          (set-node-visits! node (+ (node-visits node) 1))
          (set-node-wins! node (+ (node-wins node) result))
          (backpropagate (node-parent node) result)))))

;; -------------------------------------------------------
;; 3. MAIN EXECUTION LOOP
;; -------------------------------------------------------
(define get-next-robot
  (lambda (current-robot)
    (let* ((initial-untried (get-valid-moves current-robot))
           (root (make-node (list current-robot goal #t) '() initial-untried)))
      
      ;; Execute 3,000 MCTS cycles. This volume balances necessary 
      ;; statistical depth with manageable computation times per turn.
      (let loop ((i 0))
        (if (< i 1000)
            (let* ((selected (select-node root))
                   (expanded (expand-node selected))
                   (result (simulate expanded)))
              (backpropagate expanded result)
              (loop (+ i 1)))))
      
      ;; Decision: Discard UCB win rates and strictly pick the immediate 
      ;; physical child node that the algorithm decided to visit most often.
      (let ((children (node-children root)))
        (if (null? children)
            current-robot 
            (let loop2 ((c children) (best-child (car children)) (max-v -1))
              (if (null? c)
                  ;; Extract the physical move coordinate from the winning node state
                  (car (node-state best-child))
                  (let ((v (node-visits (car c))))
                    (if (> v max-v)
                        (loop2 (cdr c) (car c) v)
                        (loop2 (cdr c) best-child max-v))))))))))