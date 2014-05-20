(ns ch3-discovering-groups.core
  (:require [clojure.set]
            [clojure.math.numeric-tower :as math]))

(defn- get-scores [data person movies] 
  (vals (select-keys (get data person) (keys movies))))

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
         num     (- XY (/ (* X Y) (count v1)))
         n       (count v1)
         den     (math/sqrt
                   (* (- XX (/ (math/expt X 2) n)) 
                      (- YY (/ (math/expt Y 2) n))))]
    den))