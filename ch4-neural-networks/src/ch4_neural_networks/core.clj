(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn rnd [] (+ (* (- 1 -1) (rand)) -1))

(defn sigmoid [x] (Math/tanh x))

(defn dsigmoid [y] (- 1.0 (* y y)))

(defn make-network [rows cols f]
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
        -input        (if (zero? idx) input (get-in network [(dec idx) :outputs]))
        result        (forward weights -input)
        -network      (assoc-in network [idx :activations] (:activations result))]
    (assoc-in -network [idx :outputs] (:outputs result))))

(defn forward-propagate [network input]
  (let [idx-weights-coll (map-indexed #(vector %1 (:weights %2)) network)]
    (reduce #(forward-layer %1 %2 input) network idx-weights-coll)))

(defn output-error [network expected-output]
  (let [output (first (:outputs (last network)))
        error  (- expected-output output)
        delta  (* error (dsigmoid output))]
    (assoc-in network [(dec (count network)) :deltas] [delta])))

(defn transpose [coll]
  (apply map vector coll))

(defn back-prop-layer [network idx-layer parted-net]
  (let [layer->  (first parted-net)
        ->layer  (second parted-net)
        sum-errs (map #(reduce + (interleave-multiply % (:deltas ->layer))) (transpose (:weights ->layer)))
        deltas   (interleave-multiply sum-errs (map dsigmoid (:outputs layer->)))]
    (assoc-in network [idx-layer :deltas] (vec deltas))))

(defn backward-propagate [network expected-output]
  (let [-network   (output-error network expected-output)
        parted-net (vec (reverse (partition 2 1 -network)))]
    (reduce-kv #(back-prop-layer %1 %2 %3) -network parted-net)))

(defn err-deriv [deriv inputs delta]
  (let [-deriv (mapv #(+ %1 (* %2 delta)) (drop-last deriv) inputs)] 
    (conj -deriv (+ (last deriv) (* delta 1.0)))))

(defn err-deriv-layer [network idx-layer _ inputs]
  (let [-inputs (if (zero? idx-layer) inputs (get-in network [(dec idx-layer) :outputs]))
        derivs  (get-in network [idx-layer :derivs])
        -derivs (map-indexed #(err-deriv %2 -inputs (get-in network [idx-layer :deltas %1])) derivs)]
    (assoc-in network [idx-layer :derivs] (vec -derivs))))

(defn error-derivatives [network inputs]
  (let [idx-layers (vec (range (count network)))]
    (reduce-kv #(err-deriv-layer %1 %2 %3 inputs) network idx-layers)))
