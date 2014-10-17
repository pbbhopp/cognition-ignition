(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd []
  (+ (* (- 1 -1) (rand)) -1))

(defn make-neuron [num-inputs]
  {:weights    (repeat-vector (inc num-inputs) rnd)
   :last_delta (repeat-vector (inc num-inputs) (fn [] 0))
   :deriv      (repeat-vector (inc num-inputs) (fn [] 0))})

(defn make-layer [num-nodes num-inputs]
  (into [] (take num-nodes (repeatedly #(make-neuron num-inputs)))))
