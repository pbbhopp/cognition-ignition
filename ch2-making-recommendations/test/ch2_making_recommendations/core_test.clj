(ns ch2-making-recommendations.core-test
  (:require [clojure.test :refer :all]
            [ch2-making-recommendations.core :refer :all]))

(def critics {"Lisa Rose" {"Lady in the Water" 2.5, "Snakes on a Plane" 3.5,
                           "Just My Luck" 3.0, "Superman Returns" 3.5, "You, Me and Dupree" 2.5,
                           "The Night Listener" 3.0}
              "Gene Seymour" {"Lady in the Water" 3.0, "Snakes on a Plane" 3.5,
                              "Just My Luck" 1.5, "Superman Returns" 5.0, "The Night Listener" 3.0,
                              "You, Me and Dupree" 3.5}
              "Michael Phillips" {"Lady in the Water" 2.5, "Snakes on a Plane" 3.0,
                                  "Superman Returns" 3.5, "The Night Listener" 4.0}
              "Claudia Puig" {"Snakes on a Plane" 3.5, "Just My Luck" 3.0,
                              "The Night Listener" 4.5, "Superman Returns" 4.0,
                              "You, Me and Dupree" 2.5}
              "Mick LaSalle" {"Lady in the Water" 3.0, "Snakes on a Plane" 4.0,
                              "Just My Luck" 2.0, "Superman Returns" 3.0, "The Night Listener" 3.0,
                              "You, Me and Dupree" 2.0}
              "Jack Matthews" {"Lady in the Water" 3.0,"Snakes on a Plane" 4.0,
                               "The Night Listener" 3.0, "Superman Returns" 5.0, "You, Me and Dupree" 3.5}
              "Toby" {"Snakes on a Plane" 4.5, "You, Me and Dupree" 1.0, "Superman Returns" 4.0}})

(deftest sim-distance-test
  (testing "similarity scores: Euclidean distance"
    (is (= (sim-distance critics ["Lisa Rose" "Gene Seymour"]) 0.14814814814814814))))

(deftest sim-pearson-test
  (testing "similarity scores: Pearson correlation coefficient"
    (is (= (sim-pearson critics ["Lisa Rose" "Gene Seymour"]) 0.39605901719066977))))

(deftest top-matches-test
  (testing "function that scores everyone against a given person and finds the closest matches"
    (is (= (top-matches critics "Toby" :n 3) 
           [[0.9912407071619299 "Lisa Rose"] [0.9244734516419049 "Mick LaSalle"] [0.8934051474415647 "Claudia Puig"]]))))
