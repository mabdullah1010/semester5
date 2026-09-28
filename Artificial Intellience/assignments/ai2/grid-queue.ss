;;Muhammad Abdullah
;;September 19th 2026
;;AI2


(define queue '())

(define front
  (lambda ()
    (if (null? queue)
        '()
    ;else
        (car queue))))

(define dequeue
  (lambda ()
    (if (null? queue)
        '()
    ;else   
        (let ((temp (front)))
           (set! queue (cdr queue))
           temp))))

(define enqueue
  (lambda (lst)
    (set! queue (append queue lst))))


