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

(defn cover [over under]
  (into over (subvec under (count over))))

(defn transpose [coll]
  (apply map vector coll))
