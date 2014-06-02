(ns ch3-discovering-groups.core
  (:require [clojure.set]
            [clojure.math.numeric-tower :as math]))

(defn make-calc-sums []
  {:X 0 :Y 0 :XX 0 :YY 0 :XY 0})

(defn calc-xy [x y]
  {:X x :Y y :XX (math/expt x 2) :YY (math/expt y 2) :XY (* x y)})

(defn add-sums [sum1 sum2]
  (merge-with + sum1 sum2))

(defn pearson
  [v1 v2]
  (let [sums     (reduce
                   (fn [sums [x y]] (add-sums sums (calc-xy x y)))
                   (make-calc-sums)
                   (map vector v1 v2))
        {X  :X   
         Y  :Y 
         XX :XX 
         YY :YY
         XY :XY} sums
         n       (count v1)
         num     (- XY (/ (* X Y) n))
         den     (math/sqrt
                   (* (- XX (/ (math/expt X 2) n)) 
                      (- YY (/ (math/expt Y 2) n))))]
    (if (pos? den) (- 1.0 (/ num den)) 0)))

(defn find-by [f lazy-coll]
  (reduce #(conj %1 (apply f %2)) [] lazy-coll))

(defn centroids [min-max]
  (+ (* (rand) (- (second min-max) (first min-max))) (first min-max)))

(defn find-closet-centroid [point centroids]
  (let [dists   (map #(pearson % point) centroids)
        closest (apply min dists)]
    (.indexOf dists closest)))

(defn find-closet-centroids [kclusters data] 
  (map #(find-closet-centroid % kclusters) data))

(defn cols-sums [coll]
  (map #(apply + %) (partition (count coll) (apply interleave coll))))

(defn make-ranges [data]
  (let [nrows     (count data)
        col-group (partition nrows (apply interleave data))
        ranges    (partition 2 (interleave (find-by min col-group) (find-by max col-group)))]
    ranges))

(defn calc-kcluster [ranges]
  (map centroids ranges))

(defn make-kclusters [k data]
  (take k (repeatedly (fn [] (calc-kcluster (make-ranges data))))))

(defn group-by-cluster [clusters data]
  (group-by first (partition 2 (interleave clusters data))))
    
(defn mov-avg [idx kgroups]
  (let [points (map second (get kgroups idx))
        sums   (if (> (count points) 1) (cols-sums points) points)] 
    (map #(/ % (count points)) sums)))

(defn mov-avgs [kgroups kclusters]
  (map-indexed #(if (contains? kgroups %1) (mov-avg %1 kgroups) %2) kclusters))