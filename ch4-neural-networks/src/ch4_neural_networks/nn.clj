(ns ch4-neural-networks.nn)

(defrecord Neuron [weights last-delta deriv])

(defn rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn make-vector [num-inputs f] (into [] (take num-inputs (repeatedly f))))

(defn make-neuron [num-inputs]
  (->Neuron (make-vector (inc num-inputs) rnd)
            (make-vector (inc num-inputs) (fn [] 0.0))
            (make-vector (inc num-inputs) (fn [] 0.0))))

(defn activate [weights input-vector]
  (let [init (* (last weights) 1)
        coll (map * (drop-last weights) input-vector)]
    (reduce + init coll)))

(defn update-neuron [neuron input f]
  (-> neuron
      (assoc :activation (activate (:weights neuron) input))
      (assoc :output (f (:activation neuron)))))

(defn forward-propagate [nn input-vector f]
  (let [out-fn (fn [layer] (mapv :output layer))
        inputs (reduce #(conj %1 (out-fn %2)) (vector input-vector) (drop-last nn))
        lay-fn (fn [layer input f] (mapv #(update-neuron % input f) layer))
        nn     (mapv #(lay-fn %1 %2 f) nn inputs)]
    (:output (first (last nn)))))



