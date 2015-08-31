(ns ch4-neural-networks.nn
  (:require [ch4-neural-networks.layer :refer :all]))

(defrecord NN [layers learn-rate])

(defn make-layers [dims]
  (mapv #(make-layer (first %) (second %)) dims))

(defn make-nn [dims rate]
  (->NN (make-layers dims) rate))

(defn- update-layers [agg layers f]
  (reduce #(conj %1 (f %1 %2)) agg layers))

(defn forward-prop [nn inputs f]
  (let [ls  (:layers nn)
        out (feed (first ls) inputs f)
        ls  (update-layers (vector out) (rest ls) (fn [a l] (feed l (:activations (last a)) f)))]
    (assoc nn :layers ls)))

(defn train [nn x y f df]
  (let [out  (:activations (last (:layers (forward-prop nn x f))))
        err  (map - y out)
        bck  (backprop (last (:layers nn)) err df) 
        corr (update-layers (list bck) (rest (reverse (:layers nn))) (fn [a l] (backprop l (:deltas (first a)) f)))]
    (assoc nn :layers (into [] corr))))
