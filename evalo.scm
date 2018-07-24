(load "mk.scm")

;; This is an interpreter for a simple Lisp.  Variables in this language are
;; represented namelessly, using De Bruijn indices.
;; Because it is implemented as a relation, we can run this interpreter with
;; unknowns in any argument position.  If we place unknowns in the `expr`
;; position, we can synthesize programs.
(define-relation (evalo expr env value)
  (conde            ;; `conde` creates a disjunction
    ((fresh (body)  ;; `fresh` introduces new logic variables and a conjunction
                    ;; `==` constrains two terms to be equal
       (== `(lambda ,body) expr)      ;; expr is a lambda definition
       (== `(closure ,body ,env) value)))
    ((== `(quote ,value) expr))       ;; expr is a literal constant
    ((fresh (a*)
       (== `(list . ,a*) expr)
       (eval-listo a* env value)))
    ((fresh (index)
       (== `(var ,index) expr)        ;; expr is a variable
       (lookupo index env value)))
    ((fresh (rator rand arg env^ body)
       (== `(app ,rator ,rand) expr)  ;; expr is a function application
       (evalo rator env `(closure ,body ,env^))
       (evalo rand env arg)
       (evalo body `(,arg . ,env^) value)))
    ((fresh (a d va vd)
       (== `(cons ,a ,d) expr)        ;; expr is a cons operation
       (== `(,va . ,vd) value)
       (evalo a env va)
       (evalo d env vd)))
    ((fresh (c va vd)
       (== `(car ,c) expr)            ;; expr is a car operation
       (== va value)
       (evalo c env `(,va . ,vd))))
    ((fresh (c va vd)
       (== `(cdr ,c) expr)            ;; expr is a cdr operation
       (== vd value)
       (evalo c env `(,va . ,vd))))))

;; We limit valid literals to these values to simplify the implementation.
;; In practice, unlimited literals can be supported.
(define (atomo v)
  (conde
    ((== '() v))
    ((== 'a v))
    ((== #t v))
    ((== #f v))
    ((== 'b v))
    ((== '1 v))
    ((== 'x v))
    ((== 'y v))
    ((== 's v))))

;; Lookup the value a variable is bound to.
;; Variables are represented namelessly using 0-based De Bruijn indices.
;; These indices are encoded as peano numerals: (), (s), (s s), etc.
(define-relation (lookupo index env value)
  (fresh (arg e*)
    (== `(,arg . ,e*) env)
    (conde
      ((== '() index) (== arg value))
      ((fresh (i* a d)
         (== `(s . ,i*) index)
         (== `(,a . ,d) e*)
         (lookupo i* e* value))))))

;; This helper evaluates arguments to a list construction.
(define-relation (eval-listo e* env value)
  (conde
    ((== '() e*) (== '() value))
    ((fresh (ea ed va vd)
       (== `(,ea . ,ed) e*)
       (== `(,va . ,vd) value)
       (evalo ea env va)
       (eval-listo ed env vd)))))
