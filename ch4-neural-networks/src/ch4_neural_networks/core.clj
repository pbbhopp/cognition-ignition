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

(defn delta [neuron sum-error]
  (let [-neuron (assoc neuron :delta sum-error)]
    -neuron))

(defn backward-propagate
  ([->net expected-output]
    (let [net<- (reverse ->net)
          f     (fn [neuron] (* (- expected-output (:output neuron)) (dsigmoid (:output neuron))))
          layer (map #(delta % (f %)) (first net<-))
          <-net (conj [] layer)]
      (backward-propagate (rest net<-) layer <-net)))
  ([net<- next-layer <-net]
    (if (empty? net<-)
      (reverse <-net)
      (let [f     (fn [idx l] (reduce + (map #(* (get-in % [:weights idx]) (:delta %)) l)))
            layer (map-indexed #(delta %2 (* (dsigmoid (:output %2)) (f %1 next-layer))) (first net<-))
            <-net (conj <-net layer)]
        (backward-propagate (rest net<-) layer <-net)))))

