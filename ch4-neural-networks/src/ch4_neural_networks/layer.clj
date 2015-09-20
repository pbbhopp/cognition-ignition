(ns ch4-neural-networks.layer)
  
(defrecord Layer [weights activations errors deltas bias])

(defn rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn make-vector [inputs f] (into [] (take inputs (repeatedly f))))

(defn make-layer [num-neurons inputs]
  (->Layer (into [] (take num-neurons (repeatedly (partial make-vector inputs rnd))))
           (make-vector num-neurons (fn [] 0.0))
           (make-vector num-neurons (fn [] 0.0))
           (make-vector num-neurons (fn [] 0.0))
           (make-vector num-neurons (fn [] 0.0))))

(defn- inputs-with-bias [inputs bias]
  (let [offset (- (count bias) (count inputs))]
    (reduce #(update-in %1 [(+ (first %2) offset)] + (second %2)) bias (map-indexed #(vector %1 %2) inputs)))) 

(defn feed [layer inputs f]
  (let [w (:weights layer)
        x (if (nil? (:bias layer)) inputs (inputs-with-bias inputs (:bias layer)))
        v (map #(reduce + (map * x %)) w)]
    (assoc layer :activations (mapv f v))))

(defn backprop [layer in df]
  (let [w (:weights layer)
        e (mapv #(* %1 %2) in (map df (:activations layer)))
        l (assoc layer :errors e)]
    (assoc l :deltas (mapv #(reduce + (map * e %)) w))))

(defn update-weights [layer out rate]
  (let [w (:weights layer)
        e (:errors layer)
        deltas (map #(map (partial * rate %) out) e)
        w (mapv #(mapv + %1 %2) w deltas)]
    (assoc layer :weights w)))
