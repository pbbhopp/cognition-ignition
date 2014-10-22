(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd [] (+ (* (- 1 -1) (rand)) -1))

(defn sigmoid [x] (Math/tanh x))

(defn dsigmoid [y] (- 1.0 (* y y)))

(defn make-neuron [num-inputs]
  {:weights    (repeat-vector (inc num-inputs) rnd)
   :last-delta (repeat-vector (inc num-inputs) (fn [] 0))
   :deriv      (repeat-vector (inc num-inputs) (fn [] 0))})

(defn make-layer [num-nodes num-inputs]
  (into [] (take num-nodes (repeatedly #(make-neuron num-inputs)))))

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

(defn sum-errors [idx-neuron layer]
  (let [mult-coll (map #(* (get-in % [:weights idx-neuron]) (:delta %)) layer)]
    (reduce + mult-coll)))

(defn backward-propagate-layer [network idx-layer part]
  (let [layer1  (first part)
        layer2  (second part)]
    (doseq [idx (range (count layer1))]
      (swap! network assoc-in [idx-layer idx :delta] 
             (* (sum-errors idx layer2) (dsigmoid (get-in layer1 [idx :output])))))))
  
(defn backward-propagate-net [network expected-output]
  (let [_        (output-error network expected-output)
        net-part (reverse (partition 2 1 @network))]
    (doseq [idx  (range (count net-part))
            part net-part]
      (backward-propagate-layer network idx part))))

(defn error-derivatives [network input idx-layer idx-neuron]
  (doseq [idx-input (range (count input))
          signal    input]
    (swap! network update-in [idx-layer idx-neuron :deriv idx-input] 
           + (* (get-in @network [idx-layer idx-neuron :delta]) signal))) 
  (swap! network update-in [idx-layer idx-neuron :deriv (count input)] 
         + (* (get-in @network [idx-layer idx-neuron :delta]) 1.0)))  
  
(defn error-derivatives-layer [network input idx-layer]
  (let [layer  (get @network idx-layer)
        -input (if (zero? idx-layer) input (get-outputs network (dec idx-layer)))]
    (doseq [idx-neuron (range (count layer))]
      (error-derivatives network -input idx-layer idx-neuron))))
 
(defn error-derivatives-net [network input]
  (doseq [idx-layer (range (count @network))]
    (error-derivatives-layer network input idx-layer)))

(defn update-weights [network rate momentum idx-layer idx-neuron]
  (let [weights (get-in @network [idx-layer idx-neuron :weights])]
    (doseq [idx-weight (range (count weights))]
      (let [delta (+ (* rate (get-in @network [idx-layer idx-neuron :deriv idx-weight])) 
                     (* momentum (get-in @network [idx-layer idx-neuron :last-delta idx-weight])))]
        (swap! network update-in [idx-layer idx-neuron :weights idx-weight] + delta)
        (swap! network assoc-in [idx-layer idx-neuron :last-delta idx-weight] delta)
        (swap! network assoc-in [idx-layer idx-neuron :deriv idx-weight] 0.0)))))
  
(defn update-weights-layer [network rate momentum idx-layer]
  (let [layer (get @network idx-layer)]
    (doseq [idx-neuron (range (count layer))]
      (update-weights network rate momentum idx-layer idx-neuron))))

(defn update-weights-net [network rate momentum]
  (doseq [idx-layer (range (count @network))]
    (update-weights-layer network rate momentum idx-layer)))