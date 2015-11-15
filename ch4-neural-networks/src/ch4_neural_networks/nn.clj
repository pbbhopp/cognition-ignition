(ns ch4-neural-networks.nn
  (:use [clojure.pprint]))

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

(defn- update-neuron [neuron input f]
  (-> neuron
      (assoc :activation (activate (:weights neuron) input))
      (assoc :output (f (:activation neuron)))))

(defn forward-propagate [nn input-vector f]
  (let [out-fn (fn [layer] (mapv :output layer))
        inputs (reduce #(conj %1 (out-fn %2)) (vector input-vector) (drop-last nn))
        lay-fn (fn [layer input f] (mapv #(update-neuron % input f) layer))
        nn     (mapv #(lay-fn %1 %2 f) nn inputs)]
    (:output (first (last nn)))))

(defmulti update-layer (fn [nn idx ex df] (if (= (last nn) (get nn idx)) :last :other)))

(defmethod update-layer :last [nn idx ex df]
  (let [out (get-in nn [idx 0 :output])
        err (- ex out)]
    (assoc-in nn [idx 0 :delta] (* err (df out)))))

(defmethod update-layer :other [nn idx _ df]
  (let [m (fn [layer k] (map #(* (get-in % [:weights k]) (get % :delta)) layer))
        s (fn [nn k] (reduce + (m (get nn (inc idx)) k)))
        f (fn [nn k] (assoc-in nn [idx k :delta] (* (s nn k) (df (get-in nn [idx k :output])))))]
    (reduce #(f %1 %2) nn (range (count (get nn idx))))))

(defn backward-propagate [nn expected-output df]
  (let [index (reverse (range (count nn)))]
    (reduce #(update-layer %1 %2 expected-output df) nn index)))

(defn calc-err-derivatives [nn input-vector]
  (let [inputs  (concat (vector input-vector) (mapv #(mapv (fn [n] (:output n)) %) (drop-last nn)))
        inputs  (mapv #(concat % [1]) inputs)
        deriv-f (fn [deriv delta input] (mapv + (map #(* % delta) input) deriv))
        assoc-f (fn [nn l-idx n-idx]
                  (let [neuron (get-in nn [l-idx n-idx])]
                    (assoc-in nn [l-idx n-idx :deriv]
                              (deriv-f (:deriv neuron) (:delta neuron) (get inputs l-idx)))))
        layer-f (fn [nn l-idx] (reduce #(assoc-f %1 l-idx %2) nn (range (count (get nn l-idx)))))]
    (reduce #(layer-f %1 %2) nn (range (count nn)))))
