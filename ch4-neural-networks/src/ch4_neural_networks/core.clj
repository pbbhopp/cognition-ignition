(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn rnd [] (+ (* (- 1 -1) (rand)) -1))

(defn sigmoid [x] (Math/tanh x))

(defn dsigmoid [y] (- 1.0 (* y y)))

(defn make-network
  [rows cols f]
  (vec (repeat rows (vec (repeat cols f)))))

(defn make-layer [num-nodes num-inputs]
  {:weights    (make-network num-nodes num-inputs rnd)
   :last-delta (make-network num-nodes num-inputs (fn [] 0))
   :deriv      (make-network num-nodes num-inputs (fn [] 0))})

(defn interleave-multiply [& colls]
  (let [coll (partition (count colls) (apply interleave colls))]
    (map #(reduce * %) coll)))

(defn activate [weights input]
  (let [init-sum (* (last weights) 1)
        mults    (interleave-multiply (drop-last weights) input)
        sum      (reduce + mults)]
    (+ init-sum sum)))

(defn forward 
  ([weights input]
    (let [active (activate (first weights) input)
          output (sigmoid active)]
      (forward (rest weights) input (vector active) (vector output))))
  ([weights input activations outputs]
    (if (empty? weights)
      {:activations activations :outputs outputs}
      (let [active (activate (first weights) input)
            output (sigmoid active)]
        (forward (rest weights) input (conj activations active) (conj outputs output))))))

(defn forward-layer [network idx-weights input]
  (let [[idx weights] idx-weights
        -input      (if (zero? idx) input (get-in network [(dec idx) :outputs]))
        result      (forward weights -input)
        -network    (assoc-in network [idx :activations] (:activations result))]
    (assoc-in -network [idx :outputs] (:outputs result))))

(defn forward-propagate [network input]
  (let [idx-weights-coll (map-indexed #(vector %1 (:weights %2)) network)]
    (reduce #(forward-layer %1 %2 input) network idx-weights-coll)))

