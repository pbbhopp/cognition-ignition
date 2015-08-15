(ns ch4-neural-networks.layer)
  
(defrecord Layer [weights activations errors])

(defn rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn make-vector [n f] (into [] (take n (repeatedly f))))

(defn make-layer [n m]
  (->Layer (into [] (take n (repeatedly (partial make-vector m rnd))))
           (make-vector n (fn [] 0.0))
           (make-vector n (fn [] 0.0))))
           
(defn feed [w x f]
  (let [coll (map #(reduce + (map (fn [a b] (* a b)) %1 %2)) w x)]
    (map f coll)))

(defn backprop [w v s df]
  (let [err (map #(* %1 %2) s (map df v))]
    (map #(reduce + (map * err %)) w)))

