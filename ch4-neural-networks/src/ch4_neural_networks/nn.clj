(ns ch4-neural-networks.nn
  (:require [ch4-neural-networks.layer :refer :all]))

(defrecord NN [layers learn-rate])

(defn make-layers [dims]
  (mapv #(make-layer (first %) (second %)) dims))

(defn make-nn [dims rate]
  (->NN (make-layers dims) rate))

(defn forward-prop [nn x f]
  (let [ls (:layers nn)
         _ (println (:weights (first ls)))
        v  (feed (first ls) x f)] 
    (reduce #(feed %2 %1 f) v (rest ls))))
