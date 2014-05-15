(ns ch2-making-recommendations.core
  (:require [clojure.set]
            [clojure.math.numeric-tower :as math]))

(defn- squares-of-differences
  [prefs people movie]
  (math/expt (- (get-in prefs [(first people) movie]) (get-in prefs [(second people) movie])) 2))

(defn sim-distance
  "Determines how similar people are in their tastes. You do this by comparing
   each person with every other person and calculating a similarity score. 
   Calculating similarity scores: Euclidean distance."
  [prefs people]
  (let [sim-liked-movies (reduce
                           (fn [dict movie] (assoc dict movie 1)) 
                           {}
                           (first (apply clojure.set/intersection 
                             (map (fn [person] (hash-set (keys (get prefs person)))) people))))
        sum-of-squares   (reduce 
                           + 
                           (map #(squares-of-differences prefs people %) (keys sim-liked-movies)))]
    (/ 1 (+ 1 sum-of-squares))))
