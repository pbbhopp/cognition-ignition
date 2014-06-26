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

(defn calc-error-output [target outputs]
  (let [substract-coll (partition 2 (interleave target (first outputs)))
        output-diffs   (mapv #(- (first %) (second %)) substract-coll)
        mult-coll      (partition 2 (interleave output-diffs (first outputs)))]
    (mapv #(* (first %) (sigmoid-derivative (second %))) mult-coll)))

(defn calc-error-hidden [output-deltas out-weights hidden-layer]
  (let [sum-coll  (map #(interleave (first %) (second %)) (partition 2 (interleave (take 3 (repeat output-deltas)) out-weights)))
        error     (reduce (fn [sum el] (+ sum (* (first el) (second el)))) 0 sum-coll)
        mult-coll (partition 2 (interleave [error error error] out-weights))
        errors    (partition 2 (interleave (mapv #(* (first %) (first (second %))) mult-coll) (first hidden-layer)))]
    (mapv #(* (first %) (sigmoid-derivative (second %))) errors)))

(defn update-output-weights [output-deltas out-weights hidden-layer learning-rate momentum-factor]
  (let [out-delts (take (count (first hidden-layer)) (repeat (first output-deltas)))
        mult-coll (partition 2 (interleave out-delts (first hidden-layer)))
        changes   (mapv #(* (first %) (second %)) mult-coll)
        sums-coll (partition 3 (interleave (mapv #(first %) out-weights) (mapv #(* learning-rate %) changes) (mapv #(* momentum-factor %) changes)))
        sums      (mapv #(+ (first %) (second %) (nth % 2)) sums-coll)]
    (cover (subvec sums 0 (count hidden-layer)) (mapv #(first %) out-weights))))

(defn back-propagate [neural-network target learning-rate momentum-factor]
  (let [output-deltas (calc-error-output target (:output-activ @neural-network))
        hidden-deltas (calc-error-hidden output-deltas (:output-weights @neural-network) (:hidden-activ @neural-network))
        out-weights   (update-output-weights output-deltas (:output-weights @neural-network) (:hidden-activ @neural-network) learning-rate momentum-factor)]
    out-weights))

