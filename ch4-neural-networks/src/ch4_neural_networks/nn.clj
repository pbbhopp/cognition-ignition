(ns ch4-neural-networks.nn
  (:use [clojure.pprint]))

(defn dot [v w] (reduce + (map * v w)))

(defn activate [weights input-vector activate-fn] (activate-fn (dot weights input-vector)))

(defn forward-propagate [nn input-vector activate-fn]
  (let [activate-fn (fn [outputs layer]
                      (let [biased-input (conj (last outputs) 1)]
                        (for [neuron layer] (activate neuron biased-input activate-fn))))]
    (rest (reduce #(conj %1 (activate-fn %1 %2)) (vector input-vector) nn))))

(defmulti update-layer (fn [nn idx ex df] (if (= (last nn) (get nn idx)) :last :other)))

(defmethod update-layer :last [nn idx ex df]
  (let [out (get-in nn [idx 0 :output])
        err (- ex out)]
    (assoc-in nn [idx 0 :delta] (* err (df out)))))

(defmethod update-layer :other [nn idx _ df]
  (let [m (fn [layer k] (map #(* (get-in % [:weights k]) (get % :delta)) layer))
        s (fn [nn k] (reduce + (m (get nn (inc idx)) k)))
        f (fn [nn k] (assoc-in nn [idx k :delta] (* (s nn k) (df (get-in nn [idx k :output])))))]
    (reduce #(f %1 %2) nn (range (count (get nn idx))))))

(defn backward-propagate [nn expected-output df]
  (let [index (reverse (range (count nn)))]
    (reduce #(update-layer %1 %2 expected-output df) nn index)))

(defn calc-err-derivatives [nn input-vector]
  (let [inputs  (concat (vector input-vector) (mapv #(mapv (fn [n] (:output n)) %) (drop-last nn)))
        inputs  (mapv #(concat % [1]) inputs)
        deriv-f (fn [deriv delta input] (mapv + (map #(* % delta) input) deriv))
        assoc-f (fn [nn l-idx n-idx]
                  (let [neuron (get-in nn [l-idx n-idx])]
                    (assoc-in nn [l-idx n-idx :deriv]
                              (deriv-f (:deriv neuron) (:delta neuron) (get inputs l-idx)))))
        layer-f (fn [nn l-idx] (reduce #(assoc-f %1 l-idx %2) nn (range (count (get nn l-idx)))))]
    (reduce #(layer-f %1 %2) nn (range (count nn)))))

(defn update-weights [nn lr m]
  (let [delta-f (fn [neuron]
                  (map + (map #(* lr %) (:deriv neuron)) (map #(* m %) (:last-delta neuron))))
        assoc-f (fn [nn l-idx n-idx]
                  (let [neuron (get-in nn [l-idx n-idx])
                        delta  (delta-f neuron)]
                    (-> nn
                        (assoc-in [l-idx n-idx :weights] (mapv + (:weights neuron) delta))
                        (assoc-in [l-idx n-idx :last-delta] delta)
                        (assoc-in [l-idx n-idx :deriv] 0))))
        layer-f (fn [nn l-idx] (reduce #(assoc-f %1 l-idx %2) nn (range (count (get nn l-idx)))))]
    (reduce #(layer-f %1 %2) nn (range (count nn)))))

(defn train-network [nn domain lrate mom f df]
  (let [input (vec (drop-last domain))
        expected (get domain 2)
        f  (fn [net input]
             (let [_   (forward-propagate net input f)
                   net (backward-propagate net expected df)]
               (calc-err-derivatives nn input)))
        nn (reduce #(f %1 %2) nn domain)]
    (update-weights nn lrate mom)))

