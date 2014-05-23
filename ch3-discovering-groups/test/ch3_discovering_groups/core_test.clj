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

(deftest kmeans-test
  (testing "kmeans clustering"
    (is (= (kmeans k-data)
           [[1.0 5.0] [1.0 7.0]]))))
