(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd []
  (+ (* (- 1 -1) (rand)) -1))

(defn make-matrix [rows cols val-fn]
  (vec (take rows (repeatedly #(repeat-vector cols val-fn)))))

(defn make-neural-network [input-nodes hidden-nodes output-nodes]
  (atom {:hidden-nodes (make-matrix 1 (+ 1 hidden-nodes) (fn [] 1))
         :input-nodes  (make-matrix 1 (+ 1 input-nodes) (fn [] 1))
         :output-nodes (make-matrix 1 output-nodes (fn [] 1))
         :input-weights  (make-matrix input-nodes hidden-nodes rnd)
         :output-weights (make-matrix hidden-nodes output-nodes rnd)}))

(defn transpose [coll]
  (apply map vector coll))

(defn group-by-summation [coll] 
  (map #(reduce + %) coll))

(defn group-by-multiply [& colls]
  (let [coll (partition (count colls) (apply interleave colls))]
    (map #(reduce * %) coll)))

(defn sigmoid [x] (Math/tanh x))

(defn sigmoid-derivative [y] (- 1.0 (* y y)))

(defn activate-nodes [nodes weights f]
  (let [weights (transpose weights)
        mults   (map #(group-by-multiply nodes %) weights) 
        sums    (group-by-summation mults)]
    (mapv #(f %) sums)))

(defn replace-tail [with-old-tail with-new-tail]
  (into (vec (butlast with-old-tail)) (vector (last with-new-tail))))

(defn add-tail [tail-less with-tail]
  (into tail-less (vector (last with-tail))))

(defn feed-forward [neural-network input]
  (let [train  (add-tail input (first (:input-nodes @neural-network)))
        in-ws  (:input-weights @neural-network)
        out-ws (:output-weights @neural-network)]
    (swap! neural-network assoc :hidden-nodes 
      [(replace-tail (activate-nodes train in-ws sigmoid) (first (:input-nodes @neural-network)))])
    (swap! neural-network assoc :output-nodes 
      [(activate-nodes (first (:hidden-nodes @neural-network)) out-ws sigmoid)])))
