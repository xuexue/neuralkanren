;; Contents:
;; 0. Convenient shorthands
;; 1. Data model for logic programming with transparent constraint trees
;; 2. Constraint tree visualization
;; 3. Standard biased interleaving search interface
;; 4. Standard miniKanren EDSL (Embedded Domain Specific Language) definitions
;; 5. Interface for externally-driven search
;; 6. Performance testing interface for biased interleaving search with pruning

;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 0. Convenient shorthands
;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-syntax let*/and
  (syntax-rules ()
    ((_ () rest ...) (and rest ...))
    ((_ ((name expr) ne* ...) rest ...)
     (let ((name expr))
       (and name (let*/and (ne* ...) rest ...))))))

(define-syntax defrecord
  (syntax-rules ()
    ((_ name name?)
     (begin
       (define name (vector 'name))
       (define (name? datum) (eq? name datum))))
    ((_ name name? (field set-field) ...)
     (begin
       (define (name field ...) (vector 'name field ...))
       (define (name? datum)
         (and (vector? datum) (eq? 'name (vector-ref datum 0))))
       (let ()
         (define (range-assoc start xs)
           (let loop ((xs xs) (idx start))
             (if (null? xs)
               '()
               (cons (cons (car xs) idx) (loop (cdr xs) (+ idx 1))))))
         (define (define-field-getter name rassc)
           (define idx (cdr (assoc name rassc)))
           (eval `(define (,name datum) (vector-ref datum ,idx))))
         (define (define-field-setter name rassc)
           (define idx (cdr (assoc name rassc)))
           (eval `(define (,name datum value)
                    (let ((new (vector-copy datum)))
                      (vector-set! new ,idx value)
                      new))))
         (let ((fns (range-assoc 1 '(field ...))))
           (begin (define-field-getter 'field fns) ...))
         (let ((set-fns (range-assoc 1 '(set-field ...))))
           (begin (define-field-setter 'set-field set-fns) ...)))))
    ((_ name name? field ...)
     (begin
       (define (name field ...) (vector 'name field ...))
       (define (name? datum)
         (and (vector? datum) (eq? 'name (vector-ref datum 0))))
       (let ()
         (define (range-assoc start xs)
           (let loop ((xs xs) (idx start))
             (if (null? xs)
               '()
               (cons (cons (car xs) idx) (loop (cdr xs) (+ idx 1))))))
         (define (define-field-getter name rassc)
           (define idx (cdr (assoc name rassc)))
           (eval `(define (,name datum) (vector-ref datum ,idx))))
         (let ((fns (range-assoc 1 '(field ...))))
           (begin (define-field-getter 'field fns) ...)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 1. Data model for logic programming with transparent constraint trees
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

;; Read section "3. Standard biased interleaving search interface" to see how
;; a typical miniKanren search implementation maps onto this data model.

;; The full constraint tree grammar consists of:
;; * conjunctions (logical AND) with two children
(defrecord conj conj? conj-c1 conj-c2)
;; * disjunctions (logical OR) with two children
(defrecord disj disj? disj-c1 disj-c2)
;; * recursive constraints that are currently suspended
(defrecord zzz zzz? zzz-metadata zzz-wake)
;; * subtrees that have not yet propagated equality info (stored in a state)
(defrecord pause pause? pause-state pause-goal)
;; * equalities between two terms
(defrecord == ==? ==-t1 ==-t2)
;; The interaction system currently only presents constraint trees that are in
;; disjunctive normal form (DNF), and that have propagated all equality
;; information, meaning no `pause` or `==` nodes remain.

;; Logic variables
(defrecord var var? var-index)
(define var/fresh
  (let ((index -1))
    (lambda ()
      (set! index (+ 1 index))
      (var index))))
;(define var=? eq?)
(define (var=? t1 t2)
  (and (var? t1) (var? t2) (eqv? (var-index t1) (var-index t2))))
(define (var<? v1 v2) (< (var-index v1) (var-index v2)))
(define var-initial (var/fresh))

;; States describing constraint information.  Currently, only equality
;; constraints are supported, and stored as an association list of
;; variable bindings.
(define store-empty '())
(define (store-empty? store) (null? store))
(define (store-ref store key . default)
  (let ((binding (assoc key store)))
    (if binding
      (cdr binding)
      (if (null? default)
        (error 'store-ref (format "missing key ~s in ~s" key store))
        (car default)))))
(define (store-set store key value) `((,key . ,value) . ,store))

(define (vattrs-get vs vr) (store-ref vs vr vr))
(define (vattrs-set vs vr value) (store-set vs vr value))
(define (walk-vs vs tm)
  (if (var? tm)
    (let ((va (vattrs-get vs tm)))
      (if (var=? tm va)
        tm
        (walk-vs vs va)))
    tm))

(defrecord state state? (state-vs set-state-vs))
(define state-empty (state store-empty))
(define (state-empty? st) (store-empty? (state-vs st)))
(define (state-var-get st vr) (vattrs-get (state-vs st) vr))
(define (state-var-set st vr value)
  (set-state-vs st (vattrs-set (state-vs st) vr value)))

;; Unification (for implementing equality constraints)
(define (state-var-== st vr value)
  (let*/and ((st (not-occurs? st vr value)))
    (state-var-set st vr value)))
(define (state-var-==-var st v1 v2)
  (if (var<? v1 v2)  ;; Pointing new to old may yield flatter substitutions.
    (state-var-set st v2 v1)
    (state-var-set st v1 v2)))

(define (walk st tm) (walk-vs (state-vs st) tm))

(define (not-occurs? st vr tm)
  (if (pair? tm)
    (let*/and ((st (not-occurs? st vr (walk st (car tm)))))
      (not-occurs? st vr (walk st (cdr tm))))
    (and (not (var=? vr tm)) st)))

(define (unify st t1 t2)
  (and st (let ((t1 (walk st t1)) (t2 (walk st t2)))
            (cond
              ((eqv? t1 t2) st)
              ((var? t1)
               (if (var? t2)
                 (state-var-==-var st t1 t2)
                 (state-var-== st t1 t2)))
              ((var? t2) (state-var-== st t2 t1))
              ((and (pair? t1) (pair? t2))
               (let*/and ((st (unify st (car t1) (car t2))))
                 (unify st (cdr t1) (cdr t2))))
              (else #f)))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 2. Constraint tree visualization
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (store-sort vs)
  (define (odds xs)
    (cond ((null? xs) '())
          ((null? (cdr xs)) (list (car xs)))
          (else (cons (car xs) (odds (cddr xs))))))
  (define (merge xs ys)
    (cond ((null? xs) ys)
          ((null? ys) xs)
          ((<= (var-index (caar xs)) (var-index (caar ys)))
           (cons (car xs) (merge (cdr xs) ys)))
          (else (cons (car ys) (merge xs (cdr ys))))))
  (cond ((null? vs) '())
        ((null? (cdr vs)) vs)
        (else (merge (store-sort (odds vs))
                     (store-sort (odds (cdr vs)))))))

;; A state describes equality constraints, associating variables with values.
;; Since we currently flatten constraint trees into disjunctive normal form,
;; and propagate equality information, only the binding for var-initial is
;; interesting.  If we omit this flattening, bindings for other variables will
;; often be necessary to fully specify the constraint tree.  If we inccorporate
;; primitive constraints beyond equality (i.e. disequality, types, etc.), we
;; will have to provide these as well.
(define (state-pretty st)
  ;; Minimal state mode:
  `(state ((== ,(reify-var (var-index var-initial))
               ,(reify #f st var-initial))))
  ;; Full state mode:
  ;`(state ,(map (lambda (kv) `(== ,(reify-var (var-index (car kv)))
                                  ;,(reify #f st (cdr kv))))
                ;(store-sort (state-vs st))))
  )
(define (goal-pretty goal)
  (cond
    ((conj? goal)
     `(conj ,(goal-pretty (conj-c1 goal)) ,(goal-pretty (conj-c2 goal))))
    ((disj? goal) `(disj ,(goal-pretty (disj-c1 goal)) ,(goal-pretty (disj-c2 goal))))
    ((zzz? goal) (zzz-metadata goal))
    ((==? goal) `(== ,(==-t1 goal) ,(==-t2 goal)))))
(define (stream-find-state ss)
  (cond ((conj? ss) (stream-find-state (conj-c1 ss)))
        ((pause? ss) (pause-state ss))
        (else state-empty)))
(define (stream-pretty ss)
  (define (pretty ss)
    (cond
      ((conj? ss) `(conj ,(pretty (conj-c1 ss))
                         ,(reify #f (stream-find-state (conj-c1 ss))
                                 (goal-pretty (conj-c2 ss)))))
      ((disj? ss) `(disj ,(pretty (disj-c1 ss)) ,(pretty (disj-c2 ss))))
      ((pause? ss)
       `(pause ,(state-pretty (pause-state ss))
               ,(reify #f (pause-state ss) (goal-pretty (pause-goal ss)))))
      (else ss)))
  (let loop ((ss ss) (states '()))
    (cond
      ((state? ss) (loop #f (cons ss states)))
      ((pair? ss) (loop (cdr ss) (cons (car ss) states)))
      (else (list (map reify-initial (reverse states)) (pretty ss))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 3. Standard biased interleaving search interface
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define (bind ss goal)
  (cond
    ((not ss) #f)
    ((state? ss) (start ss goal))
    ((pair? ss) (disj (pause (car ss) goal) (conj (cdr ss) goal)))
    (else (conj ss goal))))
(define (mplus s1 s2)
  (cond
    ((not s1) s2)
    ((state? s1) (cons s1 s2))
    ((pair? s1) (cons (car s1) (disj s2 (cdr s1))))
    (else (disj s2 s1))))

(define (start st goal)
  (cond
    ((conj? goal) (bind (start st (conj-c1 goal)) (conj-c2 goal)))
    ((disj? goal) (disj (pause st (disj-c1 goal)) (pause st (disj-c2 goal))))
    ((zzz? goal) (start st ((zzz-wake goal))))
    ((==? goal) (unify st (==-t1 goal) (==-t2 goal)))
    (else (error 'start (format "invalid goal to start: ~s" goal)))))

(define (continue ss)
  (cond
    ((conj? ss) (bind (continue (conj-c1 ss)) (conj-c2 ss)))
    ((disj? ss) (mplus (continue (disj-c1 ss)) (disj-c2 ss)))
    ((pause? ss) (start (pause-state ss) (pause-goal ss)))
    ((not ss) #f)
    ((state? ss) (cons ss #f))
    (else (error 'start (format "invalid stream to continue: ~s" ss)))))

(define (step n ss)
  (cond
    ((= 0 n) ss)
    ((not ss) #f)
    ((pair? ss) (cons (car ss) (step n (cdr ss))))
    (else (step (- n 1) (continue ss)))))

(define (stream-next ps)
  (define ss (continue ps))
  (cond
    ((not ss) '())
    ((state? ss) (cons ss #f))
    ((pair? ss) ss)
    (else (stream-next ss))))
(define (stream-take n ps)
  (if (and n (= 0 n))
    '()
    (let ((ss (stream-next ps)))
      (if (pair? ss)
        (cons (car ss) (stream-take (and n (- n 1)) (cdr ss)))
        '()))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 4. Standard miniKanren EDSL definitions
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define-syntax define-relation
  (syntax-rules ()
    ((_ (name param ...) body ...)
     (define (name param ...)
       (zzz `(name ,param ...) (lambda () body ...))))))

(define succeed (== #t #t))
(define fail (== #f #t))

(define-syntax conj*
  (syntax-rules ()
    ((_) succeed)
    ((_ g) g)
    ((_ gs ... g-final) (conj (conj* gs ...) g-final))))
(define-syntax disj*
  (syntax-rules ()
    ((_) fail)
    ((_ g) g)
    ((_ g0 gs ...) (disj g0 (disj* gs ...)))))

(define-syntax fresh
  (syntax-rules ()
    ((_ (vr ...) g0 gs ...) (let ((vr (var/fresh)) ...) (conj* g0 gs ...)))))
(define-syntax conde
  (syntax-rules ()
    ((_ (g0 gs ...)) (conj* g0 gs ...))
    ((_ c0 c1 cs ...) (disj (conde c0) (conde c1 cs ...)))))

(define (run-goal n st goal) (stream-take n (pause st goal)))

(define (walk* st tm)
  (let ((tm (walk st tm)))
    (if (pair? tm)
      `(,(walk* st (car tm)) . ,(walk* st (cdr tm)))
      tm)))

(define (reify-var idx)
  (string->symbol (string-append "_." (number->string idx))))
(define (reify index st tm)
  (let loop
    ((rvs store-empty) (index index) (tm tm) (k (lambda (rvs i tm) tm)))
    (let ((tm (walk st tm)))
      (cond
        ((var? tm)
         (let* ((idx (store-ref rvs tm (or index (var-index tm))))
                (n (reify-var idx)))
           (if (eqv? index idx)
             (k (store-set rvs tm index) (+ 1 index) n)
             (k rvs index n))))
        ((pair? tm) (loop rvs index (car tm)
                          (lambda (r i a)
                            (loop r i (cdr tm)
                                  (lambda (r i d) (k r i `(,a . ,d)))))))
        (else (k rvs index tm))))))
(define (reify-initial st) (reify 0 st var-initial))

(define-syntax query
  (syntax-rules ()
    ((_ (vr ...) g0 gs ...)
     (let ((goal (fresh (vr ...) (== (list vr ...) var-initial) g0 gs ...)))
       (pause state-empty goal)))))
(define-syntax run
  (syntax-rules ()
    ((_ n body ...) (map reify-initial (stream-take n (query body ...))))))
(define-syntax run*
  (syntax-rules ()
    ((_ body ...) (run #f body ...))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 5. Interface for externally-driven search
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define prune-depth 50)

(define (state*->disj ss)
  (define (success st) (pause st succeed))
  (cond ((pair? ss) (disj (success (car ss)) (state*->disj (cdr ss))))
        ((state? ss) (success ss))
        (else ss)))

(define (trivial-success? ss)
  (and (==? ss) (state-empty? (unify state-empty (==-t1 ss) (==-t2 ss)))))

(define (wake-path path ss)
  (cond
    ((conj? ss) (wake-bind (wake-path path (conj-c1 ss)) (conj-c2 ss)))
    ((disj? ss)
     (cond ((null? path) (error 'wake-path (format "path is too short ~s" ss)))
           ((car path) (wake-mplus (wake-path (cdr path) (disj-c1 ss))
                                   (disj-c2 ss)))
           (else (wake-mplus (disj-c1 ss)
                             (wake-path (cdr path) (disj-c2 ss))))))
    ((pause? ss) (wake-path-goal path (pause-state ss) (pause-goal ss)))
    (else (error 'wake-path (format "cannot wake ~s ~s" path ss)))))
(define (wake-path-goal path st g)
  (cond
    ((conj? g) (wake-bind (wake-path-goal path st (conj-c1 g)) (conj-c2 g)))
    ((disj? g)
     (cond
       ((null? path) (error 'wake-path-goal (format "path is too short ~s" g)))
       ((car path) (wake-mplus (wake-path-goal (cdr path) st (disj-c1 g))
                               (pause st (disj-c2 g))))
       (else (wake-mplus (pause st (disj-c1 g))
                         (wake-path-goal (cdr path) st (disj-c2 g))))))
    ((zzz? g)
     (if (pair? path)
       (error 'wake-path-goal (format "path is too long ~s ~s" path g))
       (prune-goal prune-depth st ((zzz-wake g)))))
    ((==? g)
     (if (pair? path)
       (error 'wake-path-goal (format "path is too long ~s ~s" path g))
       (unify st (==-t1 g) (==-t2 g))))
    (else (error 'wake-path-goal (format "cannot wake ~s ~s" path g)))))
(define (wake-mplus c1 c2)
  (cond ((not c1) c2)
        ((state? c1) (cons c1 c2))
        ((pair? c1) (cons (car c1) (wake-mplus (cdr c1) c2)))
        ((not c2) c1)
        ((state? c2) (cons c2 c1))
        ((pair? c2) (cons (car c2) (wake-mplus c1 (cdr c2))))
        (else (disj c1 c2))))
(define (wake-bind ss goal)
  (cond ((not ss) #f)
        ((state? ss) (prune-goal prune-depth ss goal))
        ((pair? ss) (wake-bind (state*->disj ss) goal))
        ((disj? ss) (normalize-left #t ss goal))
        ((pause? ss) (normalize-left #t ss goal))
        (else (error 'wake-bind (format "invalid ss: ~s" ss)))))

(define (prune-parallel-goal wake? st g)
  (define (pg gx)
    (define ss (prune-goal #f st gx))
    (cond ((not ss) #f)
          ((state? ss) ss)
          ((and (pause? ss) (not (disj? (pause-goal ss))))
           (prune-parallel-goal wake? (pause-state ss) (pause-goal ss)))
          (else (pause st g))))
  (cond ((conj? g)
         (normalize-left wake? (prune-parallel-goal wake? st (conj-c1 g))
                         (conj-c2 g)))
        ((zzz? g) (if wake? (pg ((zzz-wake g))) (pause st g)))
        (else (pg g))))
(define (prune-parallel-pause wake? ss goal)
  (let ((g1 (pause-goal ss))
        (ss2 (prune-parallel-goal wake? (pause-state ss) goal)))
    (cond ((not ss2) #f)
          ((state? ss2) (pause ss2 g1))
          ((pause? ss2) (pause (pause-state ss2)
                               (normalize-left wake? g1 (pause-goal ss2))))
          (else
            (error 'prune-parallel-pause (format "invalid ss2: ~s" ss2))))))

(define (prune force? ss)
  (cond
    ((conj? ss) (prune-bind force? (prune force? (conj-c1 ss)) (conj-c2 ss)))
    ((disj? ss) (prune-mplus force? (prune #f (disj-c1 ss)) (disj-c2 ss)))
    ((pause? ss) (prune-goal force? (pause-state ss) (pause-goal ss)))
    ((not ss) #f)
    (else ss)))
(define (prune-goal force? st goal)
  (cond
    ((conj? goal)
     (prune-bind force? (prune-goal force? st (conj-c1 goal)) (conj-c2 goal)))
    ((disj? goal)
     (prune force? (disj (pause st (disj-c1 goal)) (pause st (disj-c2 goal)))))
    ((zzz? goal)
     (if force?
       (prune-goal (or (eq? #t force?) (and (< 1 force?) (- force? 1)))
                   st ((zzz-wake goal)))
       (pause st goal)))
    ((==? goal) (unify st (==-t1 goal) (==-t2 goal)))
    (else (error 'prune-goal (format "unexpected goal: ~s" ss)))))
(define (prune-mplus force? c1 c2)
  (define (build c1 c2)
    (cond ((not c1) (if force? (prune force? c2) c2))
          ((state? c1) (cons c1 c2))
          ((pair? c1) (cons (car c1) (build (cdr c1) c2)))
          ((not c2) (if force? (prune force? c1) c1))
          ((state? c2) (cons c2 c1))
          ((pair? c2) (cons (car c2) (build c1 (cdr c2))))
          (else (disj c1 c2))))
  (if c1 (let ((c2 (prune #f c2)))
           (cond (c2 (build c1 c2))
                 (force? (prune force? c1))
                 (else c1)))
    (prune force? c2)))
(define (prune-bind force? ss goal)
  (cond ((not ss) #f)
        ((state? ss) (prune-goal force? ss goal))
        ;; NOTE: using force? instead of #f in child prunes.
        ((pair? ss) (prune-bind force? (state*->disj ss) goal))
        ((disj? ss) (normalize-left force? ss goal))
        ((pause? ss) (normalize-left force? ss goal))
        (else (error 'prune-bind (format "invalid ss: ~s" ss)))))

(define (normalize-left prune? lhs g)
  (cond ((not lhs) #f)
        ((state? lhs) (prune-parallel-goal prune? lhs g))
        ((pair? lhs) (normalize-left prune? (state*->disj lhs) g))
        ((disj? lhs)
         (normalize-left-mplus (normalize-left prune? (disj-c1 lhs) g)
                               (normalize-left prune? (disj-c2 lhs) g)))

        ((conj? g)
         (normalize-left prune? (normalize-left prune? lhs (conj-c1 g))
                         (conj-c2 g)))
        ((pause? lhs) (prune-parallel-pause prune? lhs g))
        ((trivial-success? lhs) g)
        (else (conj lhs g))))
(define (normalize-left-mplus ss1 ss2)
  (cond ((not ss1) ss2)
        ((state? ss1) (cons ss1 ss2))
        ((pair? ss1) (cons (car ss1) (normalize-left-mplus (cdr ss1) ss2)))
        ((not ss2) ss1)
        ((state? ss2) (cons ss2 ss1))
        ((pair? ss2) (cons (car ss2) (normalize-left-mplus ss1 (cdr ss2))))
        (else (disj ss1 ss2))))

;; Flattening into DNF is now built into prune.
(define (clean ss) (if (disj? ss) ss (prune #t ss)))

(define (expand-path path path-expected ss0)
  (define ss (clean (wake-path path ss0)))
  (list (cond ((not ss) (error 'expand-path (format "no solution ~s" ss0)))
              ((or (state? ss) (pair? ss)) 'solved)
              ((equal? path path-expected) 'good)
              (else 'unknown))
        ss))

(define (good-path hint ss g*)
  (define (use-hint st) (unify st (==-t1 hint) (==-t2 hint)))
  (define (good-path-goal st g g*)
    (cond ((conj? g) (good-path-goal st (conj-c1 g) (cons (conj-c2 g) g*)))
          ((disj? g) (let ((p1 (good-path-goal st (disj-c1 g) g*)))
                       (if p1 (cons #t p1)
                         (let ((p2 (good-path-goal st (disj-c2 g) g*)))
                           (and p2 (cons #f p2))))))
          ((zzz? g) (and (good-path-goal st ((zzz-wake g)) g*) '()))
          ((==? g)
           (let ((st (unify st (==-t1 g) (==-t2 g))))
             (and st (or (null? g*) (good-path-goal st (car g*) (cdr g*)))
                  '())))
          (else (error 'good-path-goal (format "unexpected goal ~s" g)))))
  (cond ((conj? ss) (good-path hint (conj-c1 ss) (cons (conj-c2 ss) g*)))
        ((disj? ss) (let* ((p1 (good-path hint (disj-c1 ss) g*)))
                      (if p1 (cons #t p1)
                        (let ((p2 (good-path hint (disj-c2 ss) g*)))
                          (and p2 (cons #f p2))))))
        ((pause? ss) (let ((st (use-hint (pause-state ss))))
                       (and st (good-path-goal st (pause-goal ss) g*))))
        (else (error 'good-path (format "unexpected stream ~s" ss)))))

(define (good-paths hint ss)
  (define path (good-path hint ss '()))
  (define next (expand-path path path ss))
  (define flag (car next))
  (define ss-next (cadr next))
  (cons (cons path ss) (if (eq? 'solved flag) '() (good-paths hint ss-next))))

;; Read interact-core.scm to see how this is used.
(define (interact in show out hint ss gpath show?)
  (define (valid-path? path)
    (or (null? path)
        (and (pair? path) (or (eqv? #t (car path)) (eqv? #f (car path)))
             (valid-path? (cdr path)))))
  (define good (if gpath gpath (good-path hint ss '())))
  (when show? (show ss))
  (let ((request (in)))
    (when (not (eof-object? request))
      (cond
        ((eq? 'good-path request)
         (out (list 'good-path good))
         (interact in show out hint ss good #f))
        ((eq? 'steps-remaining request)
         (out (list 'steps-remaining (map car (good-paths hint ss))))
         (interact in show out hint ss good #f))
        ((and (pair? request) (eq? 'jump-to-steps-remaining (car request)))
         (let* ((remaining (good-paths hint ss))
                (remaining-count (length remaining))
                (drop-count (- remaining-count (cadr request)))
                (chosen (and (<= 0 drop-count)
                             (> remaining-count drop-count)
                             (list-ref remaining drop-count))))
           (when (> 0 drop-count)
             (error 'interact (format "only ~s steps remain: ~s"
                                      remaining-count (cadr request))))
           (when (<= remaining-count drop-count)
             (error 'interact "cannot jump to steps-remaining lower than 1"))
           (interact in show out hint (cdr chosen) (car chosen) #t)))
        ((and (pair? request) (valid-path? request))
         (let* ((result (expand-path request good ss))
                (flag (car result))
                (ss2 (cadr result)))
           (out (list 'follow-path flag))
           (when (not (or (eq? 'solved flag) (eq? 'fail-solved flag)))
             (interact in show out hint ss2 #f #t))))
        (else (error 'interact (format "invalid request: ~s" request)))))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;; 6. Performance testing interface for biased interleaving search with pruning
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;

(define step-count 0)
(define step-report-threshold 1000)
(define (steps-reset) (set! step-count 0))
(define (steps-inc)
  (set! step-count (+ 1 step-count))
  (when (= 0 (remainder step-count step-report-threshold))
    (display (format "~s steps taken" step-count))
    (newline)))

(define (wake-interleave ss)
  (cond
    ((conj? ss)
     (wake-interleave-bind (wake-interleave (conj-c1 ss)) (conj-c2 ss)))
    ((disj? ss)
     (wake-interleave-mplus (wake-interleave (disj-c1 ss)) (disj-c2 ss)))
    ((pause? ss) (wake-interleave-goal (pause-state ss) (pause-goal ss)))
    (else ss)))
(define (wake-interleave-goal st g)
  (cond
    ((conj? g)
     (wake-interleave-bind (wake-interleave-goal st (conj-c1 g)) (conj-c2 g)))
    ((disj? g)
     (wake-interleave-mplus (wake-interleave-goal st (disj-c1 g)) (disj-c2 g)))
    ((zzz? g) (prune-goal prune-depth st ((zzz-wake g))))
    (else (error 'wake-interleave-goal (format "cannot wake ~s" g)))))
(define (wake-interleave-mplus ss c2)
  (cond ((not ss) c2)
        ((state? ss) (cons ss c2))
        ((pair? ss) (cons (car ss) (disj c2 (cdr ss))))
        (else (disj c2 ss))))
(define (wake-interleave-bind ss goal)
  (cond ((not ss) #f)
        ((state? ss) (wake-interleave-goal ss goal))
        ((pair? ss) (wake-interleave-mplus (wake-interleave-goal (car ss) goal)
                                           (conj (cdr ss) goal)))
        (else (conj ss goal))))

(define (stream-next-prune ps)
  (define ss (wake-interleave ps))
  (steps-inc)
  (cond ((not ss) '())
        ((state? ss) (cons ss #f))
        ((pair? ss) ss)
        (else (stream-next-prune ss))))
(define (stream-take-prune n ps)
  (if (and n (= 0 n))
    '()
    (let ((ss (stream-next-prune ps)))
      (if (pair? ss)
        (cons (car ss) (stream-take-prune (and n (- n 1)) (cdr ss)))
        '()))))

