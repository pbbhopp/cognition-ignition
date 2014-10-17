(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd []
  (+ (* (- 1 -1) (rand)) -1))

(defn sigmoid [x] (Math/tanh x))

(defn dsigmoid [y] (- 1.0 (* y y)))

(defn make-neuron [num-inputs]
  {:weights    (repeat-vector (inc num-inputs) rnd)
   :last_delta (repeat-vector (inc num-inputs) (fn [] 0))
   :deriv      (repeat-vector (inc num-inputs) (fn [] 0))})

(defn make-layer [num-nodes num-inputs]
  (into [] (take num-nodes (repeatedly #(make-neuron num-inputs)))))

(defn get-inputs [layer idx default]
  (if (= idx 0)
    default
    (mapv #(:output) layer)))

(defn interleave-multiply [& colls]
  (let [coll (partition (count colls) (apply interleave colls))]
    (map #(reduce * %) coll)))

(defn activation [neuron input]
  (let [init-sum (* (last (:weights neuron)) 1)
        mults    (interleave-multiply (drop-last (:weights neuron)) input)
        sum      (reduce + mults)]
    (+ init-sum sum)))

(defn activate-neuron [neuron idx-out idx-in]
  (let [activator (activation neuron [1 1])]
    (swap! network assoc-in [idx-out idx-in :activation] activator)
    (swap! network assoc-in [idx-out idx-in :output] (sigmoid activator))))

(defn activate-neurons [layer idx]
  (keep-indexed #(activate-neuron %2 idx %1) layer))

(defn forward-propagate [network input]
  (keep-indexed #(activate-neurons %2 %1) @network))