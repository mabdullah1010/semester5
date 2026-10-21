;; AI6: FOL & unification
;; Muhammad Abdullah
;; October 21st 2026

;; This program uses first order logic predicates and forward chaining with unification 
;; to move a robot across a grid. 
;; It treats obstacles as impassable unless they are low and stable enough to climb
;; and it records the path from the start to the goal.


;; match patterns
;; identify variables (with a question mark)

(define (variable? x)
  (and (symbol? x) (char=? (string-ref (symbol->string x) 0) #\?)))

;; unify predicates and return binding list (or false)
(define (unify pat1 pat2 bindings)
  (cond ((eq? bindings #f) #f)

        ((equal? pat1 pat2) bindings)
        ((variable? pat1) (unify-var pat1 pat2 bindings))
        ((variable? pat2) (unify-var pat2 pat1 bindings))


        ((and (pair? pat1) (pair? pat2))
         (unify (cdr pat1)
                (cdr pat2)
                (unify (car pat1) (car pat2) bindings)))
        (else #f)))


(define (unify-var var val bindings)
  (let ((binding (assoc var bindings)))
    (if binding

        (unify (cdr binding) val bindings)
        (cons (cons var val) bindings))))

;; substitute
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


(define start '(0 0))
(define goal-pos '(2 2))
(define grid-max 2)

(define facts
  '(
    ;;goooaaaaaal

    (goal (2 2))

    ;; obstacle
    (obstacle (1 1))

    ;; climbable and stable obstacle
    (obstacle (0 2))
    (height (0 2) low)
    (stable (0 2) yes)

    ;; unclimbable unstable obstacle
    (obstacle (2 1))
    (height (2 1) high)
    (stable (2 1) no)
   ))


(define rules
  '(
    ;; winning condition
    ( ((visited ?p1) (goal ?p1))

      ((finished)) )

    ;; move through free spaces
    ;; checking not visited prevents moving back and forth

    ( ((visited ?p1) (adjacent ?p1 ?p2) (not obstacle ?p2) (not visited ?p2))
      ((moved ?p1 ?p2)) )

    ;; climbing and stability
    ;; only works if obstacle is low and stable
    ( ((visited ?p1) (adjacent ?p1 ?p2) (obstacle ?p2) (height ?p2 low) (stable ?p2 yes) (not visited ?p2))
      ((moved ?p1 ?p2)) )
   ))


;; generate facts

(define (in-bounds? p)
  (and (<= 0 (car p) grid-max)
       (<= 0 (cadr p) grid-max)))


(define (assert-visited! pos)
  (let ((x (car pos))
        (y (cadr pos)))

    (set! facts (cons `(visited ,pos) facts))
    (for-each (lambda (n)

                (if (in-bounds? n)
                    (set! facts (cons `(adjacent ,pos ,n) facts))))

              (list (list (+ x 1) y)
                    (list (- x 1) y)
                    (list x (+ y 1))
                    (list x (- y 1))))))


;; evaluates list of antecedents against the fact base using recursion
;; gives a list of binding sets that satisfy all conditions

(define (match-conditions conditions bindings current-facts)
  (if (null? conditions)
      (list bindings)
      (let ((first-cond (car conditions))
            (rest-conds (cdr conditions)))

        (if (eq? (car first-cond) 'not)

            (let ((substituted-pred (substitute (cdr first-cond) bindings)))
              (if (not (member substituted-pred current-facts))
                  (match-conditions rest-conds bindings current-facts)
                  '()))


            (apply append
                   (map (lambda (fact)
                          (let ((new-bindings (unify first-cond fact bindings)))
                            (if new-bindings
                                (match-conditions rest-conds new-bindings current-facts)
                                '())))

                        current-facts))))))


(define (fire-rules current-rules current-facts)
  (let loop ((rs current-rules))
    (if (null? rs)
        #f
        (let* ((rule (car rs))

               (antecedents (car rule))
               (consequent (car (cadr rule)))
               (binding-sets (match-conditions antecedents '() current-facts)))

          (let try-bindings ((bsets binding-sets))
            (if (null? bsets)

                (loop (cdr rs))
                (let ((new-fact (substitute consequent (car bsets))))
                  (if (not (member new-fact current-facts))
                      new-fact
                      (try-bindings (cdr bsets))))))))))



(define (search count)
  (cond
    ((member '(finished) facts)
     (display "Goal found! Path: ")
     (display (path-to goal-pos))
     (newline))

    ((<= count 0)
     (display "Not found -- search exhausted depth limit")
     (newline))
    (else

     (let ((new-fact (fire-rules rules facts)))
       (cond
         ((not new-fact)
          (display "Not found -- no more rules apply -- dead end")
          (newline))


         ((eq? (car new-fact) 'moved)
          (let ((from (cadr new-fact))
                (to (caddr new-fact)))
            (set! came-from (cons (cons to from) came-from))
            (assert-visited! to)
            (search (- count 1))))

         (else
          (set! facts (cons new-fact facts))
          (search (- count 1))))))))



(define came-from '())

;; follow parents back to build the path
(define (path-to p)
  (let ((entry (assoc p came-from)))
  
    (if entry
        (append (path-to (cdr entry)) (list p))
        (list p))))



(assert-visited! start)
(search 100)

