(ns ch4-neural-networks.core)

(defn repeat-vector [num val]
  (vec (take num (repeat val))))

(defn rnd [a b]
  (+ (* (- b a) (rand)) a))

(defn make-matrix [rows cols val-fn]
  (vec (take rows (repeat (repeat-vector cols val-fn)))))

(defn make-neural-network [input-nodes hidden-nodes output-nodes]
  {:hidden-activ (repeat-vector (+ 1 hidden-nodes) 1) 
   :input-activ  (repeat-vector (+ 1 input-nodes) 1)
   :output-activ (repeat-vector output-nodes 1) 
   :in-weight-diff  (make-matrix input-nodes hidden-nodes #(0)) 
   :out-weight-diff (make-matrix hidden-nodes output-nodes #(0))
   :hidden-nodes (+ 1 hidden-nodes) 
   :input-nodes  (+ 1 input-nodes) 
   :output-nodes output-nodes 
   :input-weights  (make-matrix input-nodes hidden-nodes rnd) 
   :output-weights (make-matrix hidden-nodes output-nodes rnd)})

(defn activate-input [in-act input] 
  (into input (subvec in-act (count input))))

(defn activate-inputs [neural-network inputs] 
  (let [input-activations (mapv #(activate-input (:input-activ neural-network) (first %)) inputs)]
    (assoc neural-network :input-activ input-activations)))

(def sigmoid (fn [x] (Math/tanh x)))

(def sigmoid-derivative (fn [y] (- 1.0 (* y y))))

(defn activate-hidden-node [input-node weight-node]
  (let [add-coll (partition 2 (interleave input-node weight-node))]
    (reduce + (map #(* (first %) (second %)) add-coll))))
