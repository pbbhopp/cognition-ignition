(ns ch3-discovering-groups.core-test
  (:require [clojure.test :refer :all]
            [ch3-discovering-groups.core :refer :all]))

(def v1
  [0.0 1.0 0.0 0.0 3.0 3.0 0.0 0.0 3.0 0.0 6.0 0.0 1.0 0.0 4.0 3.0 0.0 0.0 0.0 0.0 0.0 4.0 0.0])
(def v2
  [0.0 2.0 1.0 0.0 6.0 2.0 1.0 0.0 4.0 5.0 25.0 0.0 0.0 0.0 6.0 12.0 4.0 2.0 1.0 4.0 0.0 3.0 0.0])

(def k-data [[1.0 1.0] [1.5 2.0] [3.0 4.0] [5.0 7.0] [3.5 5.0] [4.5 5.0] [3.5 4.5]])

(deftest pearson-test
  (testing "Pearson score"
    (is (= (pearson v1 v2) 0.25004925261947253))))

(deftest make-ranges-test
  (testing "making ranges"
    (is (= (make-ranges [[1.0 1.0] [1.5 2.0] [3.0 4.0] [5.0 7.0] [3.5 5.0] [4.5 5.0] [3.5 4.5]]) 
           '((1.0 5.0) (1.0 7.0))))))

(deftest make-kclusters-test
  (testing "making kclusters"
    (is (= (make-kclusters 2 [[1.0 1.0] [1.5 2.0] [3.0 4.0] [5.0 7.0] [3.5 5.0] [4.5 5.0] [3.5 4.5]]) 
           '((1.9414881374080548 6.977366215048392) (1.4885417715569864 5.5302676090029905))))))