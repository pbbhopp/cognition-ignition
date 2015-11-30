(ns ch4-neural-networks.nn
  (:use [clojure.pprint]))

(defn dot [v w] (reduce + (map * v w)))

(defn transpose [m] (mapv vector m))

(defn activate [weights input-vector activate-fn] (activate-fn (dot weights input-vector)))

(defn forward-propagate [nn input-vector activate-fn]
  (let [activate-fn (fn [outputs layer]
                      (let [biased-input (concat (last outputs) [1])]
                        (for [neuron layer] (activate neuron biased-input activate-fn))))]
    (rest (reduce #(conj %1 (activate-fn %1 %2)) (vector input-vector) nn))))

(defn calc-deltas [nn outputs actuals df]
  (let [output-deltas (map #(* (df %1) (- %1 %2)) (last outputs) actuals)
        hidden-deltas (map-indexed #(* (df %2) (dot output-deltas (for [n (last nn)] (get n %1)))) (first outputs))]
    (vector hidden-deltas output-deltas)))

(defmulti update-weights (fn [layer idx inputs outputs deltas] (if (zero? idx) :first :last)))
(defmethod update-weights :last [layer _ _ outputs deltas]
  (let [deltas  (transpose (last deltas))
        outputs (transpose (concat (first outputs) [1]))]
    (mapv #(mapv (fn [n o] (- n (* (first %1) (first o)))) %2 outputs) deltas layer)))
(defmethod update-weights :first [layer _ inputs _ deltas]
  (let [deltas (transpose (first deltas))
        inputs (transpose (concat inputs [1]))]
    (mapv #(mapv (fn [n i] (- n (* (first %1) (first i)))) %2 inputs) deltas layer)))

(defn backward-propagate [nn inputs actuals f df]
  (let [indices (range (count nn))
        outputs (forward-propagate nn inputs f)
        deltas  (calc-deltas nn outputs actuals df)]
    (mapv #(update-weights %1 %2 inputs outputs deltas) nn indices)))