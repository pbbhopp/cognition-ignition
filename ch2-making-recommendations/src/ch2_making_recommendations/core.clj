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

(defn sim-pearson
  "A slightly more sophisticated way to determine the similarity between people’s inter-
   ests is to use a Pearson correlation coefficient. The correlation coefficient is a mea-
   sure of how well two sets of data fit on a straight line. The formula for this is more
   complicated than the Euclidean distance score, but it tends to give better results in
   situations where the data isn’t well normalized."
  [prefs people]
  (let [sim-liked-movies (get-similiar-likes prefs people)
        sums             (reduce
                           (fn [sums [x y]] 
                             (merge-with + sums 
                               {:x-sums x :y-sums y 
                                :x-sq-sums (math/expt x 2) :y-sq-sums (math/expt y 2) 
                                :xy-sums (* x y)}))
                           {:x-sums 0 :y-sums 0 :x-sq-sums 0 :y-sq-sums 0 :xy-sums 0}
                           (map vector (vals (select-keys 
                                               (get prefs (first people)) 
                                               (keys sim-liked-movies))) 
                                       (vals (select-keys 
                                               (get prefs (second people)) 
                                               (keys sim-liked-movies)))))
        n                (count sim-liked-movies)]
    (/ (- (get sums :xy-sums) (/ (* (get sums :x-sums) (get sums :y-sums)) n)) 
       (math/sqrt (* (- (get sums :x-sq-sums) (/ (math/expt (get sums :x-sums) 2) n)) 
                     (- (get sums :y-sq-sums) (/ (math/expt (get sums :y-sums) 2) n)))))))

(defn top-matches
  "With above similiarity functions for comparing two people, we can create a function
   that scores everyone against a given person and finds the closest matches. In this
   case, we are interested in learning which movie critics have tastes simliar to mine 
   so that we know whose advice we should take when deciding on a movie. Get an ordered 
   list of people with similar tastes to the specified person"
  [prefs person & {:keys [n similarity] :or {n 3 similarity sim-pearson}}]
  (let [sim    (fn [other] [(similarity prefs [person other]) other])
        scores (into [] (map sim (keys (dissoc prefs person))))]
    (take n (sort-by first > scores))))