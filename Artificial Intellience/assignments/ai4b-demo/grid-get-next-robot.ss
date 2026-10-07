;; Muhammad Abdullah; Abdul Rehman; Joseph Coombs
;; Robot
;; MINIMAX with ALPHA_BETA pruning
;; October 2nd 2026


;; depth limit
(define max-depth 5)

;;how far ahead the evaluator checks around obstacles
(define local-search-depth 3)

(define infinity 1000000)
(define minus-infinity -1000000)

;; PERSISTENCE FACTOR - heat map

;; this 2D memory matrix tracks how many times the robot steps on each square
;; by applying a huge penalty to previously visited squares
;; the robot is forced to move out of dead ends and U shaped walls


(define robot-heatmap '())


(define init-heatmap

  (lambda ()
    (set! robot-heatmap (make-vector num-col-row))

    (let loop ((y 0))

      (if (< y num-col-row)
          (begin
            (vector-set! robot-heatmap y (make-vector num-col-row 0))
            
            (loop (+ y 1)))))))



;; returns the number of times a coordinate has been visited
(define get-heat
  (lambda (pos)
    (if (null? robot-heatmap)
        0

        (vector-ref (vector-ref robot-heatmap (cadr pos)) (car pos)))))


;; increments visit count
(define add-heat!
  (lambda (pos)

    (if (null? robot-heatmap)
        (init-heatmap))

    (let* ((x (car pos))
           (y (cadr pos))

           (current (vector-ref (vector-ref robot-heatmap y) x)))

      (vector-set! (vector-ref robot-heatmap y) x (+ current 1)))))



(define get-next-robot
  (lambda (current-robot)

    (if (null? robot-heatmap) (init-heatmap))
    
;; append the "stay" option (current-robot) to the end

    (let* ((moves (append (adjacento current-robot) (list current-robot)))

           (best-move current-robot)
           (best-score minus-infinity)

           (alpha minus-infinity)
           (beta infinity))
      
      (let loop ((m moves) (current-alpha alpha))

        (if (null? m)
            (begin

            ;; add chosen move in the heat map memory
              (add-heat! best-move)
              best-move)
              
            (let* ((score
                     (minimax (car m) goal max-depth current-alpha beta #f))
                   
                   ;Staying is allowed but disincentivized so robot doesn't
                   ;get stuck waiting behind obstacles
                   (adjusted-score
                     (if (equal? (car m) current-robot)
                         (- score 30)
                         score)))
              (if (> adjusted-score best-score)
                  (begin
                    (set! best-score adjusted-score)
                    (set! best-move (car m))))
              ;; update alpha to prune future useless branches
              (loop (cdr m) (max current-alpha best-score))))))))


(define minimax
  (lambda (r-pos g-pos depth alpha beta is-robot-turn)
    (cond

      ;; add depth to infinity so immediate capture is better than capturing later on
      ;; robot will always choose the fastest possible capture

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
        (let loop ((m
                     (if (< (+ (abs (- (car r-pos) (car g-pos)))
                            (abs (- (cadr r-pos) (cadr g-pos)))) 2)
                     ;Robot is within one move, so goal must stop
                     (list g-pos)
                     ;Otherwise goal may stay or move
                     (cons g-pos (adjacento g-pos))))
          (min-eval infinity)
          (b beta))
         (if (null? m)
             min-eval

             (let ((eval (minimax r-pos (car m) (- depth 1) alpha b #t)))
               (let ((new-min (min min-eval eval)))
                 (let ((new-b (min b eval)))

                   (if (<= new-b alpha)
                       new-min 
                       (loop (cdr m) new-min new-b)))))))))))

(define path-distance
  (lambda (start finish)
    (let loop ((queue (list (list start 0)))
               (visited (list start)))
      (cond
        ((null? queue)
         10000)
        ((equal? (caar queue) finish)
         (cadar queue))
        (else
          (let* ((current (caar queue))
                 (distance (cadar queue))
                 (neighbors (adjacento current)))
            (let add-neighbors ((lst neighbors)
                                (new-queue (cdr queue))
                                (new-visited visited))
              (cond
                ((null? lst)
                 (loop new-queue new-visited))
                ((member (car lst) new-visited)
                 (add-neighbors (cdr lst)
                   new-queue new-visited))
                (else
                  (add-neighbors (cdr lst) (append new-queue (list (list (car lst)
                                                                     (+ distance 1))))
                    (cons (car lst) new-visited)))))))))))

(define local-best-distance
  (lambda (r-pos g-pos depth)
    (let ((here
            (+ (abs (- (car r-pos) (car g-pos)))
             (abs (- (cadr r-pos) (cadr g-pos))))))
      (if (or (<= depth 0) (= here 0))
          here
          ; else
          (let loop ((moves (adjacento r-pos))
                     (best here))
            (if (null? moves)
                best
                ; else
                (loop (cdr moves)
                  (min best (local-best-distance (car moves) g-pos (- depth 1))))))))))
          

(define evaluate
  (lambda (r-pos g-pos)
  
    (let* ((dist
             (+ (abs (- (car r-pos) (car g-pos)))
              (abs (- (cadr r-pos) (cadr g-pos)))))
           ;Best distance robot could reach within a few local moves
           (local-dist (local-best-distance
                         r-pos g-pos local-search-depth))
           (mobility (length (adjacento g-pos)))
          (heat (get-heat r-pos)))
          (+ (- 0 (+ (* 10 dist) mobility (* 20 heat)))
           
           ;Reward position that have way around nearby obstacles
           (* 10 (- dist local-dist))))))