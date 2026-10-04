;; Muhammad Abdullah


;; Depth limit for the Minimax search tree
(define max-depth 3)
(define infinity 1000000)
(define minus-infinity -1000000)

;; =======================================================
;; PERSISTENCE FACTOR (HEAT MAP)
;; =======================================================
(define robot-heatmap '())

(define init-heatmap
  (lambda ()
    (set! robot-heatmap (make-vector num-col-row))
    (let loop ((y 0))
      (if (< y num-col-row)
          (begin
            (vector-set! robot-heatmap y (make-vector num-col-row 0))
            (loop (+ y 1)))))))

(define get-heat
  (lambda (pos)
    (if (null? robot-heatmap)
        0
        (vector-ref (vector-ref robot-heatmap (cadr pos)) (car pos)))))

(define add-heat!
  (lambda (pos)
    (if (null? robot-heatmap)
        (init-heatmap))
    (let* ((x (car pos))
           (y (cadr pos))
           (current (vector-ref (vector-ref robot-heatmap y) x)))
      (vector-set! (vector-ref robot-heatmap y) x (+ current 1)))))

;; =======================================================
;; 1. MAIN DECISION FUNCTION
;; =======================================================
(define get-next-robot
  (lambda (current-robot)
    (if (null? robot-heatmap) (init-heatmap))
    
    (let* ((moves (append (adjacento current-robot) (list current-robot)))
           (best-move current-robot)
           (best-score minus-infinity)
           (alpha minus-infinity)
           (beta infinity))
      
      (let loop ((m moves) (current-alpha alpha))
        (if (null? m)
            (begin
              (add-heat! best-move)
              best-move)
              
            (let ((score (minimax (car m) goal max-depth current-alpha beta #f)))
              (if (> score best-score)
                  (begin
                    (set! best-score score)
                    (set! best-move (car m))))
              (loop (cdr m) (max current-alpha best-score))))))))

;; =======================================================
;; 2. RECURSIVE MINIMAX
;; =======================================================
(define minimax
  (lambda (r-pos g-pos depth alpha beta is-robot-turn)
    (cond
      ;; THE FIX: Add 'depth' to infinity. 
      ;; An immediate capture (depth 3) scores 1,000,003.
      ;; A delayed capture (depth 1) scores 1,000,001. 
      ;; The robot will now always choose the fastest possible capture.
      ((equal? r-pos g-pos) (+ infinity depth))
      
      ((<= depth 0) (evaluate r-pos g-pos))
      
      (is-robot-turn


      (let loop ((m (append (adjacento r-pos) (list r-pos))) (max-eval minus-infinity) (a alpha))
         (if (null? m)
             max-eval
             (let ((eval (minimax (car m) g-pos (- depth 1) a beta #f)))
               (let ((new-max (max max-eval eval)))
                 (let ((new-a (max a eval)))
                   (if (<= beta new-a)
                       new-max 
                       (loop (cdr m) new-max new-a))))))))
                       
      (else
       (let loop ((m (cons g-pos (adjacento g-pos))) (min-eval infinity) (b beta))
         (if (null? m)
             min-eval
             (let ((eval (minimax r-pos (car m) (- depth 1) alpha b #t)))
               (let ((new-min (min min-eval eval)))
                 (let ((new-b (min b eval)))
                   (if (<= new-b alpha)
                       new-min 
                       (loop (cdr m) new-min new-b)))))))))))

;; =======================================================
;; 3. THE ADAPTIVE HEURISTIC EVALUATION
;; =======================================================
(define evaluate
  (lambda (r-pos g-pos)
    (let ((dist (+ (abs (- (car r-pos) (car g-pos)))
                   (abs (- (cadr r-pos) (cadr g-pos)))))
          (mobility (length (adjacento g-pos)))
          (heat (get-heat r-pos)))
          
      (- 0 (+ (* 10 dist) mobility (* 20 heat))))))