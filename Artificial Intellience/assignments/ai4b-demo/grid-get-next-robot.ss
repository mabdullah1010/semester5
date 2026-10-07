;; =======================================================
;; AI Problem 4: Game Playing (Monte Carlo Tree Search)
;; Replaces heuristic-based Minimax with pure statistical MCTS.
;; Runs 10,000 random simulations per turn to a depth of 80 steps
;; to evaluate the best physical move without measuring distance.
;; =======================================================

;; -------------------------------------------------------
;; 1. NODE DATA STRUCTURE
;; A node is represented as a vector:
;; #(state parent children untried-moves visits wins)
;; state = (list robot-pos goal-pos is-robot-turn)
;; -------------------------------------------------------
(define make-node 
  (lambda (state parent untried-moves)
    (vector state parent '() untried-moves 0 0.0)))

(define node-state      (lambda (n) (vector-ref n 0)))
(define node-parent     (lambda (n) (vector-ref n 1)))
(define node-children   (lambda (n) (vector-ref n 2)))
(define node-untried    (lambda (n) (vector-ref n 3)))
(define node-visits     (lambda (n) (vector-ref n 4)))
(define node-wins       (lambda (n) (vector-ref n 5)))

(define set-node-children! (lambda (n c) (vector-set! n 2 c)))
(define set-node-untried!  (lambda (n u) (vector-set! n 3 u)))
(define set-node-visits!   (lambda (n v) (vector-set! n 4 v)))
(define set-node-wins!     (lambda (n w) (vector-set! n 5 w)))

;; Helper to get valid moves (append "stay in place" to adjacent open blocks)
(define get-valid-moves
  (lambda (pos)
    (append (adjacento pos) (list pos))))

;; -------------------------------------------------------
;; 2. MCTS ENGINE FUNCTIONS
;; -------------------------------------------------------

;; Calculates the Upper Confidence Bound (UCB1) formula
;; UCB1 = (Wins / Visits) + C * sqrt(ln(ParentVisits) / Visits)
(define ucb-score
  (lambda (child parent-visits)
    (let ((v (node-visits child))
          (w (node-wins child)))
      (if (= v 0)
          1000000.0 
          ;; Changed 1.414 to 0.5 to heavily exploit any path that finds a 1.0
          (+ (/ w v) (* 0.5 (sqrt (/ (log parent-visits) v))))))))



;; PHASE 1: SELECTION (Tree Traversal)
;; Walks down the tree by picking children with the highest UCB score.
;; Stops and returns a node if it has untried moves or represents a terminal state.
(define select-node
  (lambda (node)
    (cond
      ;; Terminal state (Robot and Goal share the same block)
      ((equal? (car (node-state node)) (cadr (node-state node))) node)
      ;; Node has untried moves available for expansion
      ((not (null? (node-untried node))) node)
      ;; Dead end (no children and no untried moves)
      ((null? (node-children node)) node)
      ;; Recursively select the best UCB child
      (else (select-node (get-best-ucb-child node))))))

(define get-best-ucb-child
  (lambda (node)
    (let ((children (node-children node))
          (p-visits (node-visits node)))
      (let loop ((c children) (best-child (car children)) (best-score -1.0))
        (if (null? c)
            best-child
            (let ((score (ucb-score (car c) p-visits)))
              (if (> score best-score)
                  (loop (cdr c) (car c) score)
                  (loop (cdr c) best-child best-score))))))))

;; PHASE 2: EXPANSION
;; Pops one untried move, calculates the new board state, and adds a new child node.
(define expand-node
  (lambda (node)
    (if (or (null? (node-untried node))
            (equal? (car (node-state node)) (cadr (node-state node))))
        node ;; Cannot expand a terminal or fully expanded node
        
        (let* ((move (car (node-untried node)))
               (rest-untried (cdr (node-untried node)))
               (state (node-state node))
               (r-pos (car state))
               (g-pos (cadr state))
               (is-r-turn (caddr state))
               
               ;; Generate the new board state based on whose turn it was
               (new-state (if is-r-turn
                              (list move g-pos #f)
                              (list r-pos move #t)))
                              
               ;; Generate valid moves for the next player
               (new-untried (if is-r-turn
                                (get-valid-moves g-pos)
                                (get-valid-moves move)))
                                
               (new-node (make-node new-state node new-untried)))
               
          ;; Remove the move from parent's untried list and link the new child
          (set-node-untried! node rest-untried)
          (set-node-children! node (cons new-node (node-children node)))
          new-node))))

;; PHASE 3: SIMULATION (Rollout)
;; Plays a completely random ghost game up to 80 steps.
;; Returns 1.0 for a robot win, 0.0 for a loss (hitting the step limit).
(define simulate
  (lambda (node)
    (let loop ((state (node-state node)) (depth 0))
      (let ((r-pos (car state))
            (g-pos (cadr state))
            (is-r-turn (caddr state)))
        (cond
          ;; Pure Win
          ((equal? r-pos g-pos) 1.0)   
          
          ;; Pure Loss, but depth increased to 400 to allow intersection
          ((>= depth 400) 0.0)          
          
          (is-r-turn
           (let* ((moves (get-valid-moves r-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list move g-pos #f) (+ depth 1))))
          (else
           (let* ((moves (get-valid-moves g-pos))
                  (move (list-ref moves (random (length moves)))))
             (loop (list r-pos move #t) (+ depth 1)))))))))



;; PHASE 4: BACKPROPAGATION
;; Traces the result back up to the root, updating visits and wins for every node.
(define backpropagate
  (lambda (node result)
    (if (not (null? node))
        (begin
          (set-node-visits! node (+ (node-visits node) 1))
          (set-node-wins! node (+ (node-wins node) result))
          (backpropagate (node-parent node) result)))))

;; -------------------------------------------------------
;; 3. MAIN DECISION LOOP
;; -------------------------------------------------------
(define get-next-robot
  (lambda (current-robot)
    (let* ((initial-untried (get-valid-moves current-robot))
           (root (make-node (list current-robot goal #t) '() initial-untried)))
      
      ;; Lowered from 10,000 to 4,000 to offset the computational cost of depth 400
      (let loop ((i 0))
        (if (< i 4000)
            (let* ((selected (select-node root))
                   (expanded (expand-node selected))
                   (result (simulate expanded)))
              (backpropagate expanded result)
              (loop (+ i 1)))))
      
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

                        