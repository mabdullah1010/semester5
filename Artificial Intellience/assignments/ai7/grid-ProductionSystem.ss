
(define current start)

(define search
  (lambda ()
    (move_to start)))


(define rules 
  '(
    (r1 (if (current x) (adjacent y) (not path y) (not visited y) (not obstacle y))
        (delete (current x))
        (add (path x) (delete_adjacents) (move_to y)))
    (r2 (if (current x) (adjacent y) (not visited y) (not obstacle y))
        (delete (current x) (path x))
        (add (visited x) (delete_adjacents) (backtrack_to y)))
    (r3 (if (delete_adjacents) (adjacent x))
        (delete (adjacent x)))
    (r4 (if (delete_adjacents))
        (delete (delete_adjacents)))
    (r5 (if (move_to x) (not delete_adjacents))
        (delete (move_to x))
        (execute (move_to x)))
    (r6 (if (backtrack_to x) (not delete_adjacents))
        (delete (backtrack_to x))
        (execute (backtrack_to x)))
))


(define facts '())

(define move_to
  (lambda (point)
    (set! current point)
    (draw-visited (robot-x) (robot-y))
    (set! robot current)
    (draw-moved-robot (robot-x) (robot-y))
    (if (equal? point goal)
      (display "goal")
    ;else
      (let* ((adjacents (get-adjacents point))
             (observed-facts (get-observed-facts adjacents))
             (new-facts0 (append observed-facts facts))
             (adjacent-facts (get-adjacent-facts adjacents))
             (new-facts (append adjacent-facts new-facts0)))
        (set! facts (cons (list 'current point) new-facts))
        (controller)))))

(define backtrack_to
  (lambda (point)
    (set! current point)
    (draw-visited (robot-x) (robot-y))
    (set! robot current)
    (draw-moved-robot (robot-x) (robot-y))
    (let* ((adjacents (get-adjacents point))
           (adjacent-facts (get-adjacent-facts adjacents))
           (new-facts (append adjacent-facts facts)))
      (set! facts (cons (list 'current point) new-facts))
      (controller))))

(define get-adjacents 
  (lambda (point)
    (let* ((adjacent-lst (calc-adjacents point))
           (dist-adj-lst (map add-dist adjacent-lst)))
      (map cdr (sortx (car dist-adj-lst) (cdr dist-adj-lst) '())))))
      
(define calc-adjacents 
  (lambda (point)
    (let ((x (car point))
          (y (cadr point)))
      (append
          (if (< x 1) '() (list (list (- x 1) y)))
          (if (>= x (- num-col-row 1)) '() (list (list (+ x 1) y)))
          (if (< y 1) '() (list (list x (- y 1))))
          (if (>= y (- num-col-row 1)) '() (list (list x (+ y 1))))))))

(define add-dist
  (lambda (point)
    (cons
      (+ (abs (- (car point) (car goal))) (abs (- (cadr point) (cadr goal)))) 
      point)))
  
(define sortx
  (lambda (least alist blist)
    (cond 
      ((and (null? alist) (null? blist)) (list least))
      ((null? alist)
         (cons least (sortx (car blist) (cdr blist) '())))
      ((<= (car least) (caar alist))
         (sortx least (cdr alist) (cons (car alist) blist)))
      (else
         (sortx (car alist) (cdr alist) (cons least blist))))))

(define get-observed-facts
  (lambda (point-lst)
    (if (null? point-lst)
       '()
    ;else
       (append (get-observed-facts2 (car point-lst)) 
               (get-observed-facts (cdr point-lst))))))

(define get-observed-facts2
  (lambda (point) 
    (let ((point-type (get-node grid (car point) (cadr point))))
      (cond 
        ((= point-type obstacle-stable-low)
          (list (list 'obstacle point) (list 'stable point) (list 'height point 'low)))
        ((= point-type obstacle-stable-high)
          (list (list 'obstacle point) (list 'stable point) (list 'height point 'high)))
        ((= point-type obstacle-unstable-low)
          (list (list 'obstacle point) (list 'height point 'low)))
        ((= point-type obstacle-unstable-high)
          (list (list 'obstacle point) (list 'height point 'high)))
        (else
          '())))))

(define get-adjacent-facts
  (lambda (adjacents)
    (if (null? adjacents)
      '()
    ;else
      (cons (list 'adjacent (car adjacents)) (get-adjacent-facts (cdr adjacents))))))
 
(define controller
  (lambda ()
    (let ((rule (get-rule-to-fire rules)))  ; returns (rx binding)
       (cond 
         ((null? rule)
           '())
         (else
           (fire-rule rule)
           (controller))))))

(define get-rule-to-fire 
  (lambda (rule-list)
    (if (null? rule-list)
       '()
    ;else
       (let* ((rule (car rule-list))
              (rule-name (car rule))
              (rule-if (cdadr rule))
              (binding (get-binding rule-if facts '())))
          (if (not binding)
            (get-rule-to-fire (cdr rule-list))
          ;else
            (list rule-name binding))))))

(define get-pos 
  (lambda (rule-if)
     (if (null? rule-if)
       '()
     ;else
       (let ((pred (car rule-if)))
          (if (equal? 'not (car pred))
             '()
          ;else
             (cons pred (get-pos (cdr rule-if))))))))

(define get-neg
  (lambda (rule-if)
    (member (assoc 'not rule-if) rule-if)))


(define get-binding
  (lambda (predicates fact-list binding)
    (if (null? predicates)
      (cons '(bindings) binding)
    ;else
      (get-first-pred-binding (car predicates) (cdr predicates) facts binding))))

(define get-first-pred-binding
  (lambda (first-pred remaining-pred fact-list binding)
    (cond 
       ((null? fact-list)
           #f)
       ((equal? 'not (car first-pred))
           (and (not (get-first-pred-binding (cdr first-pred) '() facts binding))
                (get-binding remaining-pred facts binding)))
       (else
         (let ((bind (unify first-pred (car fact-list) binding)))
            (if bind
               (let* ((bind2 (append bind binding))
                      (final-bind (get-binding remaining-pred facts bind2)))
                  (or final-bind
                      (get-first-pred-binding first-pred remaining-pred 
                                              (cdr fact-list) binding)))
            ;else
               (get-first-pred-binding first-pred remaining-pred 
                                       (cdr fact-list) binding)))))))


(define check-neg 
  (lambda (preds binding)
    (cond 
       ((or (not preds) (null? preds))
          binding)
       ((member (binder (cdar preds) binding) facts)
          '())
       (else
          (check-neg (cdr preds) binding)))))
              
(define unify
  (lambda (pred1 pred2 bindings)
    (unify2 (binder pred1 bindings) (binder pred2 bindings) '())))
   
(define unify2
  (lambda (pred1 pred2 bind)
    (cond 
      ((and (null? pred1) (null? pred2))
         bind)
      ((or (null? pred1) (null? pred2))
         #f)
      ((equal? pred1 pred2)
         bind)
      ((equal? (car pred1) (car pred2))
         (unify2 (cdr pred1) (cdr pred2) bind))
      ((and (list? (car pred1)) (list? (car pred2))) 
         #f)
      ((list? (car pred2))
         (unify2 (cdr pred1) (cdr pred2) (cons (list (car pred1) (car pred2)) bind)))
      (else
         #f))))
      
(define fire-rule 
  (lambda (rule-binding)
    (let* ((rule-with-name (assoc (car rule-binding) rules))
           (rule (cdr rule-with-name))
           (bindings (cadr rule-binding))
           (deletes (assoc 'delete rule))
           (adds (assoc 'add rule))
           (executes (assoc 'execute rule)))
      (if deletes
         (deletex (cdr (binder deletes bindings))))
      (if adds
         (add (cdr (binder adds bindings))))
      (if executes
         (execute (cdr (binder executes bindings)))))))

(define binder
  (lambda (lst bindings)
    (if (null? lst)
      '()
    ;else
      (let ((element (car lst)))
        (cond 
          ((null? element)
            '())
          ((null? bindings)
            lst)
          ((null? (car bindings))
            lst)
          ((list? element)
            (cons (binder element bindings) (binder (cdr lst) bindings)))
          (else
            (let ((bind (assoc element bindings)))
              (if bind
                 (cons (cadr bind) (binder (cdr lst) bindings))
              ;else
                 (cons element (binder (cdr lst) bindings))))))))))

(define deletex
  (lambda (predicates)
    (set! facts (delete2 predicates facts))))

(define delete2
  (lambda (predicates fact-list)
    (if (null? fact-list)
       '()
    ;else
       (let ((fact (car fact-list)))
         (if (member fact predicates)
           (delete2 predicates (cdr fact-list))
         ;else
           (cons fact (delete2 predicates (cdr fact-list))))))))

(define add
  (lambda (predicates)
    (set! facts (add2 (reverse predicates)))))

(define add2
  (lambda (predicates)
    (if (null? predicates)
       facts
    ;else
       (let ((pred (car predicates)))
         (if (member pred facts)
           (add2 (cdr predicates))
         ;else
           (cons pred (add2 (cdr predicates))))))))

(define execute
  (lambda (predicates)
    (if (not (null? predicates))
       (let* ((pred (car predicates))
              (name (car pred))
              (point (cadr pred)))
         (pause pause-num)
         (cond
           ((equal? 'move_to name)
              (move_to point))
           ((equal? 'backtrack_to name)
              (backtrack_to point))
           (else
              (display "error in execute")))
         (execute (cdr predicates))))))

