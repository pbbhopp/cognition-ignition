(ns ch4-neural-networks.core
  (:require [clojure.math.numeric-tower :as math]))

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd []
  (+ (* (- 1 -1) (rand)) -1))

(defn make-neuron [num-inputs]
  {:weights    (repeat-vector (inc num-inputs) rnd)
   :last_delta (repeat-vector (inc num-inputs) (fn [] 0))
   :deriv      (repeat-vector (inc num-inputs) (fn [] 0))})

(defn make-layer [num-nodes num-inputs]
  (into [] (take num-nodes (repeatedly #(make-neuron num-inputs)))))

(defn transpose [coll]
  (apply map vector coll))

(defn sum-by-group [coll]
  (map #(reduce + %) coll))

(defn group-by-multiply [& colls]
  (let [coll (partition (count colls) (apply interleave colls))]
    (map #(reduce * %) coll)))

(defn sigmoid [x] (Math/tanh x))

(defn dsigmoid [y] (- 1.0 (* y y)))

(defn activate-nodes [nodes weights f]
  (let [weights (transpose weights)
        mults   (map #(group-by-multiply nodes %) weights)
        sums    (sum-by-group mults)]
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

(defn errors-for-nodes [nodes error-factors]
  (let [coll (partition 2 (interleave nodes error-factors))]
    (map #(* (dsigmoid (first %)) (second %)) coll)))

(defn update-weights [nodes weights errors rate]
  (let [changes (map #(map (fn [el] (* el %)) errors) nodes)
        rates   (map #(map (fn [el] (* el rate)) %) changes)
        coll    (map #(partition 2 %) (apply map interleave [rates weights]))]
    (map #(map (fn [el] (+ (first el) (second el))) %) coll)))

(defn vec-colls-in-coll [coll]
   (vec (map vec coll)))

(defn cover [over under]
  (into over (subvec under (count over))))

(defn back-propagate [neural-network train learning-rate]
  (let [output-nodes   (first (:output-nodes @neural-network))
        output-deltas  (apply map - [(second train) output-nodes])
        output-errors  (errors-for-nodes output-nodes output-deltas)
        hidden-deltas  (map #(group-by-multiply output-errors %) (:output-weights @neural-network))
        hidden-nodes   (first (:hidden-nodes @neural-network))
        hidden-errors  (errors-for-nodes hidden-nodes (map #(first %) hidden-deltas))
        output-weights (:output-weights @neural-network)
        input-nodes    (first (:input-nodes @neural-network))
        input-weights  (:input-weights @neural-network)
        in-weights     (update-weights hidden-nodes output-weights output-errors learning-rate)
        input-nodes    (cover (first train) input-nodes)
        out-weights    (update-weights input-nodes input-weights hidden-errors learning-rate)]
    (swap! neural-network assoc :input-weights (vec-colls-in-coll in-weights))
    (swap! neural-network assoc :output-weights (vec-colls-in-coll out-weights))))

(defn train [neural-network training-data learning-rate]
  (doseq [train training-data]
    (feed-forward neural-network (first train))
    (back-propagate neural-network train learning-rate)))
