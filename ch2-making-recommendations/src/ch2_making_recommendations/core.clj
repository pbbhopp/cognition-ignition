(ns ch2-making-recommendations.core
  (:require [clojure.set]
            [clojure.math.numeric-tower :as math]))

(defn- squares-of-differences
  [prefs people movie]
  (math/expt (- (get-in prefs [(first people) movie]) (get-in prefs [(second people) movie])) 2))

(defn- get-similiar-likes
  [prefs people]
  (reduce
    (fn [dict movie] (assoc dict movie 1)) 
    {}
    (apply clojure.set/intersection 
      (map (fn [person] (apply hash-set (keys (get prefs person)))) people))))

(defn sim-distance
  "Determines how similar people are in their tastes. You do this by comparing
   each person with every other person and calculating a similarity score. 
   Calculating similarity scores: Euclidean distance."
  [prefs people]
  (let [sim-liked-movies (get-similiar-likes prefs people)
        sum-of-squares   (reduce 
                           + 
                           (map #(squares-of-differences prefs people %) (keys sim-liked-movies)))]
    (/ 1 (+ 1 sum-of-squares))))

(defn- get-scores [data person movies] 
  (vals (select-keys (get data person) (keys movies))))

(defn make-calc-sums []
  {:X 0 :Y 0 :XX 0 :YY 0 :XY 0})

(defn calc-xy [x y]
  {:X x :Y y :XX (math/expt x 2) :YY (math/expt y 2) :XY (* x y)})

(defn add-sums [sum1 sum2]
  (merge-with + sum1 sum2))

(defn sim-pearson
  "A slightly more sophisticated way to determine the similarity between people’s inter-
   ests is to use a Pearson correlation coefficient. The correlation coefficient is a mea-
   sure of how well two sets of data fit on a straight line. The formula for this is more
   complicated than the Euclidean distance score, but it tends to give better results in
   situations where the data isn’t well normalized."
  [prefs people]
  (let [sim-liked-movies (get-similiar-likes prefs people)
        sums             (reduce
                           (fn [sums [x y]] (add-sums sums (calc-xy x y)))
                           (make-calc-sums)
                           (map 
                             vector 
                             (get-scores prefs (first people) sim-liked-movies)
                             (get-scores prefs (second people) sim-liked-movies)))
        { X :X   Y :Y 
         XX :XX YY :YY
         XY :XY}         sums
        n                (count sim-liked-movies)]
    (/ (- XY (/ (* X Y) n)) (math/sqrt (* (- XX (/ (math/expt X 2) n)) (- YY (/ (math/expt Y 2) n)))))))

(defn top-matches
  "With above similiarity functions for comparing two people, we can create a 
   function that scores everyone against a given person and finds the closest 
   matches. In this case, we are interested in learning which movie critics 
   have tastes simliar to specified person so that we know whose advice we 
   should take when deciding on a movie. Get an ordered list of people with 
   similar tastes to the specified person"
  [prefs person & {:keys [n similarity] :or {n 3 similarity sim-pearson}}]
  (let [sim    (fn [other] [(similarity prefs [person other]) other])
        scores (into [] (map sim (keys (dissoc prefs person))))]
    (take n (sort-by first > scores))))


(defn get-recommendations
  "Gets recommendations for a person by using a weighted average of every other 
   user's rankings"
  [prefs person]
  {})