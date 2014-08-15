(ns ch6-naive-bayes.core-test
  (:require [clojure.test :refer :all]
            [ch6-naive-bayes.core :refer :all]))

(def classifier (make-classifier))

(train classifier (get-words "Nobody owns water") :good)
(train classifier (get-words "quick rabbit jumps fences") :good)
(train classifier (get-words "buy pharmaceuticals now") :bad)
(train classifier (get-words "make quick money online casino") :bad)
(train classifier (get-words "quick brown fox jumps") :good)
(println @classifier)

(deftest feature-count-test
  (testing "should track feature counts correctly"
    (is (= (:quick (:data @classifier)) {:good 2 :bad 1}))
    (is (= (:jumps (:data @classifier)) {:good 2}))))

(deftest feature-probability-test
  (testing "should calculate feature probability correctly"
    (is (= (feature-probability classifier "quick" :good) (/ 2 3)))
    (is (= (feature-probability classifier "quick" :bad) (/ 1 2)))
    (is (= (feature-probability classifier "rabbit" :good) (/ 1 3)))
    (is (= (feature-probability classifier "rabbit" :bad) 0))))

(deftest category-probability-test
  (testing "should calculate category probability correctly"
    (is (= (category-probability classifier :good) (/ 3 5)))
    (is (= (category-probability classifier :bad) (/ 2 5)))))

(deftest probability-of-category-given-features-test
  (testing "should calculate conditional probability of category given some features correctly"
    (is (= (prob-of-category-given-features classifier :good [:quick :rabbit]) 0.15624999999999997))
    (is (= (prob-of-category-given-features classifier :bad [:quick :rabbit]) 0.05))))

(deftest classify-documents-test
  (testing "should classify document of features correctly"
    (is (= (classify classifier "rabbit quick") :good))
    (is (= (classify classifier "quick money") :bad))
    (is (= (classify classifier "brown rabbit") :good))
    (is (= (classify classifier "buy casino") :bad))
    (is (= (classify classifier "Nobody buy brown water") :good))
    (is (= (classify classifier "now owns online pharmaceuticals") :bad))))
