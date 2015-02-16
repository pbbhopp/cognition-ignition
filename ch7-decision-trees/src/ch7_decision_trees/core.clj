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

(defn transpose [coll]
  (apply map vector coll))
    
(defn find-best-feature-to-split [data-set]
  (let [entropy    (shannon-entropy data-set)
        feat-list  (drop-last (transpose data-set))
        fn-prob    (fn [idx value] 
                     (let [sub-data-set (split-data-set data-set idx value)]
                       (* (/ (count sub-data-set) (count data-set)) (shannon-entropy sub-data-set))))
        fn-entrop  (fn [idx feats] 
                     (reduce #(+ %1 (fn-prob idx %2)) 0.0 feats))
        info-gains (map-indexed #(vector %1 (- entropy (fn-entrop %1 %2))) feat-list)
        best-feat  (first (last (sort-by second info-gains)))]
    best-feat)) 