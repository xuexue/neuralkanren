(load "evalo.scm")

;; q-transform/hint is a convenient way to produce program synthesis queries.
;; Use this to provide the initial problem state to the Scheme interactor.
;; e.g. (q-transform/hint (quote (lambda (cdr (cdr (var ())))))
;;        (quote ((() y . 1)
;;                (#f y () . #t)
;;                (#f b () b . y)
;;                (x #f (#f . #f) . #t)
;;                (a #f y x s . a))))
;; To see the expected outputs for each example in such a query, provide the
;; query to query-outputs.scm.
(define (q-transform/hint fcode inputs)
  (define outputs (q-transform-outputs/hint fcode inputs))
  (q/hint fcode inputs outputs))

(define (q/hint fcode is os)
  (define q
    (query (defn)
      (fresh (body) (== `(lambda ,body) defn)
        (let loop ((is (reverse is)) (os (reverse os)))
          (if (null? (cdr is))
            (evalo `(app ,defn ',(car is)) '() (car os))
            (conj (loop (cdr is) (cdr os))
                  (evalo `(app ,defn ',(car is)) '() (car os))))))))
  (list (== (list fcode) var-initial) q))

(define (q-transform-outputs/hint fcode inputs)
  (car (car (run 1 (outputs)
              (evalo `(list . ,(map (lambda (i) `(app ,fcode ',i))
                                    inputs))
                     '()
                     outputs)))))
