(ns ch6-naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [ch6-naive-bayes.core :refer :all]))

(def classifier (atom {}))

(train classifier (get-words "Nobody owns water") :good)
(train classifier (get-words "quick rabbit jumps fences") :good)
(train classifier (get-words "buy pharmaceuticals now") :bad)
(train classifier (get-words "make quick money online casino") :bad)
(train classifier (get-words "quick brown fox jumps") :good)
(println @classifier)

(deftest feature-count-test
  (testing "should track feature counts correctly"
    (is (= (:quick @classifier) {:good 2 :bad 1}))
    (is (= (:jumps @classifier) {:good 2}))))

(deftest feature-probability-test
  (testing "should calculate feature probability correctly"
    (is (= (feature-probability classifier "quick" :good) (/ 2 3)))))

(deftest category-probability-test
  (testing "should calculate category probability correctly"
    (is (= (category-probability classifier :good) (/ 11 19)))
    (is (= (category-probability classifier :bad) (/ 8 19)))))

