(load "query.scm")

;; To use this interactor, first supply a query/hint, then input commands.

;; Supported command patterns:
;; good-path:       Show expansion path of next ground-truth step.
;; steps-remaining: List expansion paths of remaining ground-truth steps.
;; ((or 0 1) ...):  Follow path to recursive constraint and expand it.
;;   A path is a list of binary digits, where 0 means left and 1 means right.
(define (in)
  (define request (read))
  (cond
    ((or (eq? 'good-path request)
         (eq? 'steps-remaining request)
         (and (pair? request)
              (eq? 'jump-to-steps-remaining (car request))))
     request)
    ((pair? request)
     (map (lambda (n)
            (cond
              ((= 0 n) #t)
              ((= 1 n) #f)
              (else (error 'in (format "invalid path segment ~s" n)))))
          request))
    ((eof-object? request) (exit))
    (else (error 'in (format "unexpected request: ~s" request)))))

(define (show ss)
  (printf "~s\n" (cadr (stream-pretty ss)))
  ;; Use pretty-print instead of printf for saner debugging.
  ;(pretty-print (cadr (stream-pretty ss)))
  )

(define (out response)
  (define (bool->bit b) (if b 0 1))
  (define (bools->bits bs) (map bool->bit bs))
  (define output
    (cond
      ((eq? 'good-path (car response)) (bools->bits (cadr response)))
      ((eq? 'follow-path (car response)) (cadr response))
      ((eq? 'steps-remaining (car response)) (map bools->bits (cadr response)))
      (else (error 'out (format "unrecognized output: ~s" response)))))
  (printf "~s\n" output))

(define (read-query/hint) (eval (read)))

(define ss/hint (read-query/hint))
(define hint (car ss/hint))
(define ss (cadr ss/hint))

;; See the definition in transparent.scm for details.
(interact in show out hint (clean ss) #f #t)
