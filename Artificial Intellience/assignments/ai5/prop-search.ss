;; AI5
;; Muhammad Abdullah
;; October 11th 2026
;; 3x3 grid

(define rules
  '(
    ;; Goal verification rules
    ((visited00 goal00) (finished))
    ((visited01 goal01) (finished))
    ((visited02 goal02) (finished))
    ((visited10 goal10) (finished))
    ((visited11 goal11) (finished))
    ((visited12 goal12) (finished))
    ((visited20 goal20) (finished))
    ((visited21 goal21) (finished))
    ((visited22 goal22) (finished))

    ;; we can visit a tile if an adjacent tile is visited 
    ;; and the target tile is not an obstacle

    ((visited00 -obstacle01) (visited01))
    ((visited00 -obstacle10) (visited10))
    
    ((visited01 -obstacle00) (visited00))
    ((visited01 -obstacle02) (visited02))
    ((visited01 -obstacle11) (visited11))
    
    ((visited02 -obstacle01) (visited01))
    ((visited02 -obstacle12) (visited12))
    
    ((visited10 -obstacle00) (visited00))
    ((visited10 -obstacle20) (visited20))
    ((visited10 -obstacle11) (visited11))
    
    ((visited11 -obstacle10) (visited10))
    ((visited11 -obstacle12) (visited12))
    ((visited11 -obstacle01) (visited01))
    ((visited11 -obstacle21) (visited21))
    
    ((visited12 -obstacle11) (visited11))
    ((visited12 -obstacle02) (visited02))
    ((visited12 -obstacle22) (visited22))
    
    ((visited20 -obstacle10) (visited10))
    ((visited20 -obstacle21) (visited21))
    
    ((visited21 -obstacle20) (visited20))
    ((visited21 -obstacle22) (visited22))
    ((visited21 -obstacle11) (visited11))
    
    ((visited22 -obstacle21) (visited21))
    ((visited22 -obstacle12) (visited12))
  ))

(define facts
  '(goal22 visited00 obstacle11 obstacle12 obstacle21))

(define ModusPonens
  (lambda (rule)
    ;; if the consequent is already a fact, return '()
    (if (member (car (cadr rule)) facts)
        '()
        (ModusPonens2 (car rule) (cadr rule)))))

(define ModusPonens2 
  (lambda (b a)
    (if (null? b)
        a
        (let* ((p (car b))
               (str-p (symbol->string p)))
          ;; if the symbol represents a negative condition

          (if (char=? (string-ref str-p 0) #\-)
              (let ((pos-fact (string->symbol (substring str-p 1 (string-length str-p)))))
                (if (not (member pos-fact facts))
                    (ModusPonens2 (cdr b) a)
                    '()))

              (if (member p facts)
                  (ModusPonens2 (cdr b) a)
                  '()))))))

(define search
  (lambda (count)
    (cond
      ((member 'finished facts)
       (display "goal found")
       (newline))
      ((<= count 0)
       (display "not found")
       (newline))
       
      (else
       (let* ((firstRule (car rules))
              (remainingRules (append (cdr rules) (list firstRule)))
              (newFact (ModusPonens firstRule)))
         
         ;;rotate the rules
         (set! rules remainingRules)
         
         (cond 
           ((not (null? newFact))
            ;; new fact established and add it to the facts
            (set! facts (append newFact facts))
            (search (- count 1)))

           (else
            ;;no new fact derived
            (search (- count 1)))))))))

(search 100)