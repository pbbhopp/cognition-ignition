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
   :last-delta (repeat-vector (inc num-inputs) (fn [] 0))
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

(defn get-outputs [network idx-layer]
  (let [layer (get @network idx-layer)]
    (map #(:output %) layer)))

(defn activation [neuron input]
  (let [init-sum (* (last (:weights neuron)) 1)
        mults    (interleave-multiply (drop-last (:weights neuron)) input)
        sum      (reduce + mults)]
    (+ init-sum sum)))

(defn activate-neuron [network input idx-layer idx-neuron]
  (let [neuron    (get-in @network [idx-layer idx-neuron])
        activator (activation neuron input)]
    (swap! network assoc-in [idx-layer idx-neuron :activation] activator)
    (swap! network assoc-in [idx-layer idx-neuron :output] (sigmoid activator))))

(defn forward-propagate-layer [network input idx-layer]
  (let [layer  (get @network idx-layer)
        -input (if (zero? idx-layer) input (get-outputs network (dec idx-layer)))]
    (doseq [idx-neuron (range (count layer))]
      (activate-neuron network -input idx-layer idx-neuron))))

(defn forward-propagate-net [network input]
  (doseq [idx-layer (range (count @network))]
    (forward-propagate-layer network input idx-layer)))

(defn output-error [network expected-output]
  (let [neuron (first (last @network))
        error  (- expected-output (:output neuron))
        delta  (* error (dsigmoid (:output neuron)))]
    (swap! network assoc-in [(dec (count @network)) 0 :delta] delta)))
