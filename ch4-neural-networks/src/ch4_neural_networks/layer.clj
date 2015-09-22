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
  (let [offset (- (count bias) (count inputs))
        add-fn (fn [bias [idx v]] (update-in bias [(+ offset idx)] + v))]
    (->> (map-indexed #(vector %1 %2) inputs) 
         (reduce #(add-fn %1 %2) bias)))) 

(defn feed [layer inputs f]
  (let [{b :bias w :weights} layer
        x (if (nil? b) inputs (inputs-with-bias inputs b))
        v (map #(reduce + (map * x %)) w)]
    (assoc layer :activations (mapv f v))))

(defn backprop [layer in df]
  (let [w (:weights layer)
        e (mapv #(* %1 %2) in (map df (:activations layer)))
        l (assoc layer :errors e)]
    (assoc l :deltas (mapv #(reduce + (map * e %)) w))))

(defn update-weights [layer out rate]
  (let [{w :weights e :errors} layer
        deltas (map #(map (partial * rate %) out) e)
        w (mapv #(mapv + %1 %2) w deltas)]
    (assoc layer :weights w)))
