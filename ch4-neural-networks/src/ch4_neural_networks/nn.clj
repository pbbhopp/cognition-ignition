(ns ch4-neural-networks.nn
  (:require [ch4-neural-networks.layer :refer :all]))

(defrecord NN [layers learn-rate])

(defn make-layers [dims]
  (mapv #(make-layer (first %) (second %)) dims))

(defn make-nn [dims rate]
  (->NN (make-layers dims) rate))

(defn- update-layers [agg layers f]
  (reduce #(conj %1 (f %1 %2)) agg layers))

(defn- feed-layer-fn [activation-fn]
  (fn [updated-layers layer] 
    (feed layer (:activations (last updated-layers)) activation-fn)))

(defn forward-prop [nn inputs f]
  (let [ls  (:layers nn)
        out (feed (first ls) inputs f)
        ls  (update-layers (vector out) (rest ls) (feed-layer-fn f))]
    (assoc nn :layers ls)))

(defn- backprop-fn [derivative-fn]
  (fn [updated-layers layer] 
    (backprop layer (:deltas (first updated-layers)) derivative-fn)))

(defn train [nn x y f df]
  (let [nn   (forward-prop nn x f)
        out  (:activations (last (:layers nn)))
        err  (map - y out)
	bck  (backprop (last (:layers nn)) err df) 
	corr (update-layers (list bck) (rest (reverse (:layers nn))) (backprop-fn df))]
    (assoc nn :layers (into [] corr))))
