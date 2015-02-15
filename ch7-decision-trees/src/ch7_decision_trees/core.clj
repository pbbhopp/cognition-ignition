(ns ch7-decision-trees.core)

(defn log2 [n]
  (/ (Math/log n) (Math/log 2)))

(defn shannon-entropy [data-set]
  (let [num-entries  (count data-set)
        label-counts (frequencies (map last data-set))
        probs        (map #(/ (second %) num-entries) label-counts)
        entropy      (reduce #(- %1 (* %2 (log2 %2))) 0.0 probs)]
    entropy))

(let [data [[1 1 "yes"] [1 1 "yes"] [1 0 "no"] [0 1 "no"] [0 1 "no"]]]
  (shannon-entropy data))
