;;Muhammad Abdullah
;;September 19th 2026
;;AI2

(define num-col-row 40)
(define pause-num 200000)
(define size (floor (/ 700 num-col-row)))
(define obstacle-density 30)
(load "grid-class.ss")
(load "grid-draw.ss")
(load "grid-make.ss")
(load "grid-stack.ss")
(load "grid-queue.ss")

(define grid0 (make-grid num-col-row)) 
(draw-obstacles grid0)
(define grid (convert-grid grid0))
(load "grid-new.ss")
(load "grid-Astar.ss")
(set-goal grid)
(set-start grid)
(draw-start)
(draw-goal)
(draw-robot)
(show canvas)
(search-astar grid 20000)