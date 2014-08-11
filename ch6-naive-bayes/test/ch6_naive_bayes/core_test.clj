(ns ch6-naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [ch6-naive-bayes.core :refer :all]))

(def classifier (atom {}))

(train classifier (get-words "Nobody owns the water") :good)
(train classifier (get-words "the quick rabbit jumps fences") :good)
(train classifier (get-words "buy pharmaceuticals now") :bad)
(train classifier (get-words "make quick money at the online casino") :bad)
(train classifier (get-words "the quick brown fox jumps") :good)
(println @classifier)

(deftest feature-count-test
  (testing "should track feature counts correctly"
    (is (= (:quick @classifier) {:good 2 :bad 1}))
    (is (= (:jumps @classifier) {:good 2}))))

(deftest feature-count-test
  (testing "should calculate feature probability correctly"
    (is (= (feature-probability classifier "quick" :good) (/ 2 3)))))
