(load "query.scm")

;; USAGE EXAMPLE:
;; scheme --script query-outputs.scm < data/test_problems.txt

;; This will show the expected example outputs for all (q-transform/hint ...)
;; queries provided to stdin.

(let loop ()
  (define q (read))
  (when (eof-object? q) (exit))
  (let* ((code (eval (cadr q))) (i* (eval (caddr q))))
    (define output (q-transform-outputs/hint code i*))
    ;; Uncomment these to see the corresponding code and input portions.
    ;(printf "code: ~s\n" code)
    ;(printf "input*: ~s\n" i*)
    (printf "~s\n" output)
    (loop)))

