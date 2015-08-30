(ns ch4-neural-networks.layer)
  
(defrecord Layer [weights activations errors deltas])

(defn rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn make-vector [inputs f] (into [] (take inputs (repeatedly f))))

(defn make-layer [num-neurons inputs]
  (->Layer (into [] (take num-neurons (repeatedly (partial make-vector inputs rnd))))
           (make-vector num-neurons (fn [] 0.0))
           (make-vector num-neurons (fn [] 0.0))
           (make-vector num-neurons (fn [] 0.0))))
           
(defn feed [layer inputs f]
  (let [w    (:weights layer)
        coll (map #(reduce + (map * inputs %)) w)]
    (assoc layer :activations (map f coll))))

(defn backprop [layer errs df]
  (let [w   (:weights layer)
        e (map #(* %1 %2) errs (map df (:activations layer)))]
    (assoc layer :deltas (map #(reduce + (map * e %)) w))))

(defn update-weights [layer out rate]
  (let [w (:weights layer)
        e (:errors layer)
        deltas (map #(map (partial * rate %) out) e)
        w (mapv #(mapv + %1 %2) w deltas)]
    (assoc layer :weights w)))
