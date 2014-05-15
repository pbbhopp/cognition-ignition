(ns ch2-making-recommendations.core
  [:require clojure.set])

(defn sim-distance
  "Determines how similar people are in their tastes. You do this by comparing
   each person with every other person and calculating a similarity score. 
   Calculating similarity scores: Euclidean distance."
  [prefs people]
  (reduce
    (fn [dict movie] (assoc dict movie 1)) 
    {}
    (first (apply clojure.set/intersection 
      (map (fn [person] (hash-set (keys (get prefs person)))) people))))
)
