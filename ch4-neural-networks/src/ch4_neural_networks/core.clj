(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn rnd [] (+ (* (- 1.0 -1.0) (rand)) -1.0))

(defn sigmoid [x] (/ 1.0 (+ 1.0 (Math/exp (* -1.0 x)))))

(defn dsigmoid [y] (* y (- 1.0 y)))

(defn make-vector-with [n f]
  (take n (repeatedly f)))

(defn make-neuron [num-inputs]
  {:weights    (make-vector-with num-inputs rnd)
   :last-delta (make-vector-with num-inputs (fn [] 0))
   :deriv      (make-vector-with num-inputs (fn [] 0))})

(defn activate [weights input]
  (let [sum (reduce + (map #(* %1 %2) (drop-last weights) input))]
   (+ (* (last weights) 1) sum)))

(defn activate-neuron [neuron input]
  (let [n  (assoc neuron :activation (activate (:weights neuron) input))
        -n (assoc n :output (sigmoid (:activation n)))]
    -n))

(defn forward-propagate
  ([network input]
    (let [layer (map #(activate-neuron % input) (first network))
          input (map :output layer)
          net   (conj [] layer)]
      (forward-propagate (rest network) input net)))
  ([layers input net]
    (if (empty? layers)
      net
      (let [layer (map #(activate-neuron % input) (first layers))
            input (map :output layer)
            net   (conj net layer)]
        (forward-propagate (rest layers) input net)))))
