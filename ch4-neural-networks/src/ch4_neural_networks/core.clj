(ns ch4-neural-networks.core)

(defn repeat-vector [n val-fn]
  (vec (take n (repeatedly val-fn))))

(defn rnd []
  (+ (* (- 1 -1) (rand)) -1))

(defn make-matrix [rows cols val-fn]
  (vec (take rows (repeatedly #(repeat-vector cols val-fn)))))

(defn make-neural-network [input-nodes hidden-nodes output-nodes]
  (atom {:hidden-activ (make-matrix 1 (+ 1 hidden-nodes) (fn [] 1))
         :input-activ  (make-matrix 1 (+ 1 input-nodes) (fn [] 1))
         :output-activ (make-matrix 1 output-nodes (fn [] 1))
         :in-weight-diff  (make-matrix input-nodes hidden-nodes (fn [] 0))
         :out-weight-diff (make-matrix hidden-nodes output-nodes (fn [] 0))
         :hidden-nodes (+ 1 hidden-nodes)
         :input-nodes  (+ 1 input-nodes)
         :output-nodes output-nodes
         :input-weights  (make-matrix input-nodes hidden-nodes rnd)
         :output-weights (make-matrix hidden-nodes output-nodes rnd)}))

(defn cover [over under]
  (into over (subvec under (count over))))

(defn activate-input [in-act input]
  (cover input (first in-act)))

(defn activate-inputs [neural-network inputs]
  (mapv #(activate-input (:input-activ @neural-network) (first %)) inputs))

(defn sigmoid [x] (Math/tanh x))

(defn sigmoid-derivative [y] (- 1.0 (* y y)))

(defn activate-node [input-node weight-node act-fn]
  (let [add-coll (partition 2 (interleave (first input-node) weight-node))
        total    (reduce + (map #(* (first %) (second %)) add-coll))]
    (act-fn total)))

(defn activate-nodes [node-layer nodes act-fn]
  (let [weight-nodes (partition (count nodes) (apply interleave nodes))]
    (mapv #(activate-node node-layer % act-fn) weight-nodes)))

(defn update [neural-network train-data]
  (let [in-act  (:input-activ @neural-network)
        hid-act (:hidden-activ @neural-network)
        out-act (:output-activ @neural-network)
        wi      (:input-weights @neural-network)
        wo      (:output-weights @neural-network)]
    (swap! neural-network assoc :input-activ [(activate-input in-act (first (first train-data)))])
    (swap! neural-network assoc :hidden-activ [(cover (activate-nodes in-act (mapv drop-last wi) sigmoid) (first hid-act))])
    (swap! neural-network assoc :output-activ [(activate-nodes (:hidden-activ @neural-network) wo (fn [x] x))])))
