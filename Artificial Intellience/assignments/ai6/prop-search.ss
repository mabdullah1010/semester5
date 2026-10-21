;; =======================================================
;; AI Problem 6: First-Order Logic & Unification
;; =======================================================

;; -------------------------------------------------------
;; 1. UNIFICATION ENGINE (Pattern Matching)
;; -------------------------------------------------------

;; Identifies variables (symbols starting with '?')
(define (variable? x)
  (and (symbol? x) (char=? (string-ref (symbol->string x) 0) #\?)))

;; Unifies two predicates, returning a binding list (e.g., ((?p1 . (0 0))) ) or #f
(define (unify pat1 pat2 bindings)
  (cond ((eq? bindings #f) #f)
        ((equal? pat1 pat2) bindings)
        ((variable? pat1) (unify-var pat1 pat2 bindings))
        ((variable? pat2) (unify-var pat2 pat1 bindings))
        ((and (pair? pat1) (pair? pat2))
         (unify (cdr pat1) (cdr pat2) (unify (car pat1) (car pat2 bindings))))
        (else #f)))

(define (unify-var var val bindings)
  (let ((binding (assoc var bindings)))
    (if binding
        (unify (cdr binding) val bindings)
        (cons (cons var val) bindings))))

;; Substitutes bound variables into a predicate using the binding list
(define (substitute pat bindings)
  (cond ((variable? pat)
         (let ((binding (assoc pat bindings)))
           (if binding
               (substitute (cdr binding) bindings)
               pat)))
        ((pair? pat)
         (cons (substitute (car pat) bindings)
               (substitute (cdr pat) bindings)))
        (else pat)))

;; -------------------------------------------------------
;; 2. THE FACTS & RULES (Knowledge Base)
;; -------------------------------------------------------

(define facts
  '(
    ;; Initial starting position and Goal
    (visited (0 0))
    (goal (2 2))
    
    ;; Initial adjacencies for the starting position
    (adjacent (0 0) (0 1))
    (adjacent (0 0) (1 0))

    ;; Standard Obstacles
    (obstacle (1 1))
    
    ;; A Short, climbable obstacle
    (obstacle (0 2))
    (height (0 2) low)
    (stable (0 2) yes)
    
    ;; A Tall, unstable obstacle
    (obstacle (2 1))
    (height (2 1) high)
    (stable (2 1) no)
   ))

(define rules
  '(
    ;; RULE 1: Standard Movement (Free Spaces)
    ;; Prevents bouncing back and forth by ensuring (not visited ?p2)
    ( ((visited ?p1) (adjacent ?p1 ?p2) (not obstacle ?p2) (not visited ?p2)) 
      ((visited ?p2)) )

    ;; RULE 2: Climbing Movement (Height and Stability)
    ;; Overrides standard obstacle rule if the block is low and stable.
    ( ((visited ?p1) (adjacent ?p1 ?p2) (obstacle ?p2) (height ?p2 low) (stable ?p2 yes) (not visited ?p2)) 
      ((visited ?p2)) )

    ;; RULE 3: Victory Condition
    ( ((visited ?p1) (goal ?p1)) 
      ((finished)) )
   ))

;; -------------------------------------------------------
;; 3. DYNAMIC FACT GENERATION
;; -------------------------------------------------------

;; Purely mathematical function that asserts a tile is visited 
;; AND generates the mathematical adjacent coordinates on the fly.
(define (assert-visited! pos)
  (let ((x (car pos))
        (y (cadr pos)))
    (set! facts (cons `(visited ,pos) facts))
    ;; Generate and inject all 4 directional neighbors into the knowledge base
    (set! facts (cons `(adjacent ,pos ,(list (+ x 1) y)) facts))
    (set! facts (cons `(adjacent ,pos ,(list (- x 1) y)) facts))
    (set! facts (cons `(adjacent ,pos ,(list x (+ y 1))) facts))
    (set! facts (cons `(adjacent ,pos ,(list x (- y 1))) facts))))

;; -------------------------------------------------------
;; 4. INFERENCE ENGINE (Forward Chaining)
;; -------------------------------------------------------

;; Recursively evaluates a list of antecedents against the fact base.
;; Returns a list of valid binding sets that satisfy all conditions.
(define (match-conditions conditions bindings facts)
  (if (null? conditions)
      (list bindings)
      (let ((first-cond (car conditions))
            (rest-conds (cdr conditions)))
            
        (if (eq? (car first-cond) 'not)
            ;; CLOSED-WORLD ASSUMPTION: If we search the facts for the substituted 
            ;; predicate and do NOT find it, consider the 'not' condition satisfied.
            (let ((substituted-pred (substitute (cadr first-cond) bindings)))
              (if (not (member substituted-pred facts))
                  (match-conditions rest-conds bindings facts)
                  '()))
                  
            ;; STANDARD UNIFICATION: Search facts, unify, and branch for all matches.
            (apply append
                   (map (lambda (fact)
                          (let ((new-bindings (unify first-cond fact bindings)))
                            (if new-bindings
                                (match-conditions rest-conds new-bindings facts)
                                '())))
                        facts))))))

;; Scans through the rule base exactly once, looking for the first rule 
;; that successfully produces a brand new fact.
(define (fire-rules current-rules current-facts)
  (let loop ((rs current-rules))
    (if (null? rs)
        #f ;; No rules fired
        (let* ((rule (car rs))
               (antecedents (car rule))
               (consequent (car (cadr rule)))
               (binding-sets (match-conditions antecedents '() current-facts)))
               
          (let try-bindings ((bsets binding-sets))
            (if (null? bsets)
                (loop (cdr rs)) ;; Move to next rule
                (let ((new-fact (substitute consequent (car bsets))))
                  (if (not (member new-fact current-facts))
                      new-fact ;; Success! Return the new fact
                      (try-bindings (cdr bsets))))))))))

;; -------------------------------------------------------
;; 5. MAIN SEARCH LOOP
;; -------------------------------------------------------

(define (search count)
  (cond
    ((member '(finished) facts)
     (display "Goal found! The robot successfully navigated the environment.")
     (newline))
    ((<= count 0)
     (display "Not found. Search exhausted its depth limit.")
     (newline))
    (else
     (let ((new-fact (fire-rules rules facts)))
       (if new-fact
           (begin
             ;; Intercept 'visited' facts to dynamically generate adjacencies
             (if (eq? (car new-fact) 'visited)
                 (assert-visited! (cadr new-fact))
                 (set! facts (cons new-fact facts)))
             (search (- count 1)))
           (begin
             (display "Not found. No more rules apply (Dead End).")
             (newline)))))))

;; To run:
(search 100)