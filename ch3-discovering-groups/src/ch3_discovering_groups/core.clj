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
  (reduce #(conj %1 (apply f (doall %2))) [] lazy-coll))

(defn centroids [coll]
  (map #(+ (* (rand) (- (second %) (first %))) (first %)) coll))

(defn find-closet-centroid [matches point centroids] 
  (let [dists   (map #(pearson % point) centroids)
        closest (apply min dists) 
        idx-k   (.indexOf dists closest)]
    (merge-with into matches {idx-k #{point}})))

(defn cols-sums [coll]
  (map #(apply + %) (partition (count coll) (apply interleave coll))))

(defn mov-avgs [best-fit kclusters]
  (reduce
    (fn [coll fits]
      (let [idx  (first fits)
            pts  (second fits) 
            sums (cols-sums pts)]
        (assoc kclusters idx sums)))
    kclusters
    best-fit)
  kclusters)

(defn kmeans
  "The K-means algorithm will determine the size of the clusters based on the
   structure of the data. K-means clustering begins with k randomly placed
   centroids (points in space that represent the center of the cluster), and
   assigns every item to the nearest one. After the assignment, the centroids
   are moved to the average location of all the nodes assigned to them, and
   the assignments are redone. This process repeats until the assignments stop
   changing."
  [rows & {:keys [k iter] :or {k 2 iter 100}}]
  (let [nrows     (count rows)
        col-group (partition nrows (apply interleave rows))
        ranges    (split-at 2 (interleave (find-by min col-group) (find-by max col-group)))
        kclusters (for [i (range k)] (centroids ranges))
        matches   (reduce #(merge %1 {%2 #{}}) {} (range k)) 
        best-fit  (for [i (range iter)
                      :let [bestmatches (reduce
                                          #(find-closet-centroid %1 %2 kclusters)
                                          matches
                                          rows)
                            lastmatches bestmatches
                            kclusters   (mov-avgs bestmatches kclusters)]
                      :when (not= lastmatches bestmatches)]
                    bestmatches)]
    best-fit))