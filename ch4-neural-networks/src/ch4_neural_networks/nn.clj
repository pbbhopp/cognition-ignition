(ns ch4-neural-networks.nn
  (:require [ch4-neural-networks.layer :refer :all]))

(defrecord NN [layers learn-rate])

(defn make-layers [dims]
  (mapv #(make-layer (first %) (second %)) dims))

(defn make-nn [dims rate]
  (->NN (make-layers dims) rate))

(defn forward-prop [nn inputs f]
  (let [ls  (:layers nn)
        out (feed (first ls) inputs f)
        ls  (reduce #(conj %1 (feed %2 (:activations (last %1)) f)) [out] (rest ls))]
    (assoc nn :layers ls)))

(defn train [nn x y f df]
  (let [out  (:activations (last (:layers (forward-prop nn x f))))
        err  (map - y out)
        bck  (backprop (last (:layers nn)) err df) 
        corr (reduce #(conj %1 (backprop %2 (:deltas (first %1)) f)) '(bck) (reverse (drop-last (:layers nn))))]
    (assoc nn :layers (into [] corr))))
