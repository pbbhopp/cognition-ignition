(ns ch7-decision-trees.core)

(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))

(defn shannon-entropy [data-set]
  (let [num-entries  (count data-set)
        label-counts (frequencies (map last data-set))
        probs        (map #(/ (second %) num-entries) label-counts)
        entropy      (reduce #(- %1 (* %2 (log2 %2))) 0.0 probs)]
    entropy))

(defn split-data-set [data-set axis value]
  (let [data (filter #(= (get % axis) value) data-set)
        data (map #(concat (subvec % 0 axis) (subvec % (inc axis))) data)]
    data))  