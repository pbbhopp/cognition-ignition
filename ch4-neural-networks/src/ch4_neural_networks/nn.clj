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

