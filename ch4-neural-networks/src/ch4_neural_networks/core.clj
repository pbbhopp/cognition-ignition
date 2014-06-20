(ns ch4-neural-networks.core)

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd [a b]
  (+ (* (- b a) (rand)) a))

(defn make-matrix [rows cols val-fn]
  (vec (take rows (repeat (repeat-vector cols val-fn)))))

(defn make-neural-network [input-nodes hidden-nodes output-nodes]
  (atom {:hidden-activ (repeat-vector (+ 1 hidden-nodes) 1) 
         :input-activ  (repeat-vector (+ 1 input-nodes) 1)
         :output-activ (repeat-vector output-nodes 1) 
         :in-weight-diff  (make-matrix input-nodes hidden-nodes #(0)) 
         :out-weight-diff (make-matrix hidden-nodes output-nodes #(0))
         :hidden-nodes (+ 1 hidden-nodes) 
         :input-nodes  (+ 1 input-nodes) 
         :output-nodes output-nodes 
         :input-weights  (make-matrix input-nodes hidden-nodes #(rnd -1 1)) 
         :output-weights (make-matrix hidden-nodes output-nodes #(rnd -1 1))}))

(defn activate-input [in-act input] 
  (into input (subvec in-act (count input))))

(defn activate-inputs [neural-network inputs] 
  (mapv #(activate-input (:input-activ @neural-network) (first %)) inputs))

(defn sigmoid [x] (Math/tanh x))

(defn sigmoid-derivative [y] (- 1.0 (* y y)))

(defn activate-node [input-node weight-node]
  (let [add-coll (partition 2 (interleave input-node weight-node))
        total    (reduce + (map #(* (first %) (second %)) add-coll))]
    (sigmoid total)))

(defn activate-nodes [node-layer nodes]
  (let [weight-nodes (partition (count nodes) (apply interleave nodes))]
    (mapv #(activate-node node-layer %) weight-nodes)))

(defn update [neural-network train-data]
  (let [in-act  (:input-activ @neural-network)
        hid-act (:hidden-activ @neural-network)
        out-act (:output-activ @neural-network)
        wi      (:input-weights @neural-network)
        wo      (:output-weights @neural-network)]
    (swap! neural-network assoc :input-activ (activate-input in-act (last (first train-data))))
    (swap! neural-network assoc :hidden-activ (activate-nodes hid-act wi))
    (swap! neural-network assoc :output-activ-activ (activate-nodes out-act wo))))
